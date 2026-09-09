from io import BytesIO
import pytest
from sqlalchemy import select, event
from fyp_data_platform import ui, lifecycle
from fyp_data_platform.db.models import ModelRun, Recording, StoredFile, Annotation, TemporalWindow, Prediction
from fyp_data_platform.storage.local import LocalStorage

@pytest.fixture
def managed(settings, migrated_db, monkeypatch, tmp_path):
    engine, factory = migrated_db
    @event.listens_for(engine, 'checkout')
    def enforce_foreign_keys(connection, *_):
        connection.execute('PRAGMA foreign_keys=ON')
    storage = LocalStorage(settings.local_storage_root)
    monkeypatch.setattr(ui, 'load_settings', lambda _: settings)
    monkeypatch.setattr(ui, 'create_engine_and_session', lambda _: (engine, factory))
    monkeypatch.setattr(ui, 'create_storage', lambda _: storage)
    client = ui.create_app().test_client()
    for identity in ('r1', 'r2'):
        response = client.post('/api/uploads', data={'mode':'video', 'dataset_id':'test', 'patient_id':'p1', 'recording_id':identity, 'videos':(BytesIO(b'video'), 'video.mp4')})
        assert response.status_code == 201
    original = tmp_path / 'original.mp4'
    original.write_bytes(b'untouched source')
    for run_id in ('run1', 'run2'):
        key = f'datasets/test/derived/r1/model-runs/{run_id}/metrics.json'
        meta = storage.put_file(original, key)
        path = settings.work_root / 'test' / 'r1' / run_id
        path.mkdir(parents=True)
        (path/'input.mp4').write_bytes(b'working copy')
        with factory.begin() as session:
            session.add(ModelRun(id=run_id, name=run_id, status='completed', config_json={'dataset_id':'test','recording_id':'r1'}, metrics_json={}))
            session.add(StoredFile(dataset_id='test', recording_id='test::r1', role='model-output', modality='multimodal', source_relative_path=key, storage_backend='local', storage_key=key, payload_state='stored', size_bytes=meta.size_bytes, metadata_json={'run_id':run_id}))
    with factory.begin() as session:
        source = session.scalar(select(StoredFile).where(StoredFile.recording_id == 'test::r1', StoredFile.modality == 'video'))
        session.add(Annotation(annotation_key='truth',dataset_id='test',recording_id='test::r1',patient_id='test::p1',start_sec=0,end_sec=1,label='seizure',source_file_id=source.id))
        session.add(TemporalWindow(window_id='window1',dataset_id='test',recording_id='test::r1',patient_id='test::p1',start_sec=0,end_sec=1,feature_file_id=source.id))
        session.flush()
        session.add(Prediction(model_run_id='run1',window_id='window1',predicted_label='seizure'))
    return client, factory, storage, original

def test_archive_restore_preserves_individual_run_state(managed):
    client, factory, storage, original = managed
    assert client.patch('/api/manage/runs/run1', json={'archived':True}).status_code == 200
    assert client.patch('/api/manage/recordings/test::r1', json={'archived':True}).status_code == 200
    assert [r['recording_source_id'] for r in client.get('/api/catalog').json['recordings']] == ['r2']
    assert client.get('/api/catalog').json['runs'] == []
    assert len(client.get('/api/catalog?archived=true').json['runs']) == 2
    assert client.post('/api/runs', json={'dataset_id':'test','recording_id':'r1'}).status_code == 409
    assert client.post('/api/uploads', data={'recording_id':'r1','dataset_id':'test','videos':(BytesIO(b'new'), 'video.mp4')}).status_code == 400
    assert client.patch('/api/manage/recordings/test::r1', json={'archived':False}).status_code == 200
    assert [r['id'] for r in client.get('/api/catalog').json['runs']] == ['run2']
    assert original.read_bytes() == b'untouched source'

def test_delete_run_only_removes_its_outputs(managed, settings):
    client, factory, storage, original = managed
    assert client.delete('/api/manage/runs/run1', json={'confirmation':'run1'}).status_code == 409
    client.patch('/api/manage/runs/run1', json={'archived':True})
    assert client.delete('/api/manage/runs/run1', json={'confirmation':'wrong'}).status_code == 409
    assert client.delete('/api/manage/runs/run1', json={'confirmation':'run1'}).status_code == 200
    with factory() as session:
        assert session.get(ModelRun,'run1') is None
        assert session.get(ModelRun,'run2') is not None
        assert session.get(Recording,'test::r1') is not None
    assert not (settings.work_root/'test/r1/run1').exists()
    assert (settings.work_root/'test/r1/run2/input.mp4').exists()
    assert original.exists()

def test_recording_delete_failure_can_retry(managed, settings, monkeypatch):
    client, factory, storage, original = managed
    client.patch('/api/manage/recordings/test::r1', json={'archived':True})
    real_delete = storage.delete
    calls = []
    def broken(key):
        calls.append(key)
        if len(calls) == 2:
            raise OSError('temporary storage outage')
        real_delete(key)
    monkeypatch.setattr(storage,'delete',broken)
    assert client.delete('/api/manage/recordings/test::r1', json={'confirmation':'test::r1'}).status_code == 500
    assert client.patch('/api/manage/recordings/test::r1', json={'archived':False}).status_code == 409
    with factory() as session:
        assert session.get(Recording,'test::r1').deleting
    monkeypatch.setattr(storage,'delete',real_delete)
    assert client.delete('/api/manage/recordings/test::r1', json={'confirmation':'test::r1'}).status_code == 200
    with factory() as session:
        assert session.get(Recording,'test::r1') is None
        assert session.get(Recording,'test::r2') is not None
        assert not session.scalars(select(ModelRun)).all()
        assert len(session.scalars(select(StoredFile)).all()) == 1
        assert not session.scalars(select(Annotation)).all()
        assert not session.scalars(select(TemporalWindow)).all()
        assert not session.scalars(select(Prediction)).all()
    assert not (settings.work_root/'test/r1').exists()
    assert original.read_bytes() == b'untouched source'

def test_active_run_blocks_archive_and_delete(managed):
    client, factory, _, _ = managed
    with factory.begin() as session:
        session.get(ModelRun,'run1').status = 'running'
    assert client.patch('/api/manage/recordings/test::r1', json={'archived':True}).status_code == 409
    assert client.patch('/api/manage/runs/run1', json={'archived':True}).status_code == 409

def test_work_path_rejects_traversal(settings):
    run = ModelRun(id='run1', config_json={'dataset_id':'test', 'recording_id':'../outside'})
    with pytest.raises(ValueError):
        lifecycle._work_path(settings,'runs',run)

def test_result_exposes_pose_artifact_without_loading_it(managed):
    client, factory, _, _ = managed
    with factory.begin() as session:
        session.get(ModelRun,'run1').metrics_json = {'artifacts':[{'relative_path':'video_only/pose_keypoints.json','key':'pose.json'}]}
    response = client.get('/api/runs/run1')
    assert response.status_code == 200
    assert response.json['source_video']['pose_url'] == '/api/runs/run1/artifacts/video_only/pose_keypoints.json'
