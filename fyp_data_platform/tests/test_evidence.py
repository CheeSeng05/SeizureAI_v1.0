from types import SimpleNamespace
import numpy as np
import pytest
from fyp_data_platform.evidence import waveform, input_path, evidence_sources


def test_waveform_keeps_gap_and_maps_samples_to_recording_clock(monkeypatch):
    from seizure_multimodal.eeg import loaders
    calls = []
    class Recording:
        metadata = SimpleNamespace(channel_names=('F3',), sampling_rates=(10,), signal_units=('uV',), duration_sec=4, extra={'timeline_segments': [
            {'global_start_sec': 0, 'global_end_sec': 2, 'source_sample_start': 0, 'source_sample_stop': 20},
            {'global_start_sec': 5, 'global_end_sec': 7, 'source_sample_start': 20, 'source_sample_stop': 40}]})
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read_samples(self, **kwargs):
            calls.append(kwargs)
            return np.arange(kwargs['start_sample'], kwargs['stop_sample'], dtype=float)[None, :]
    monkeypatch.setattr(loaders, 'load_eeg', lambda _: Recording())
    data = waveform('unused', 1, 6, 0)
    assert [item['start_sample'] for item in calls] == [10, 20]
    assert data['traces'][0][0][0] == 1
    assert data['traces'][1][0][0] == 5
    assert data['traces'][1][-1][0] == 5.9
    assert waveform('unused', 2, 5, 0)['traces'] == []
    for start, end in [(0, 31), (-1, 2), (2, 1), (float('nan'), 2)]:
        with pytest.raises(ValueError): waveform('unused', start, end, 0)


def test_evidence_only_uses_verified_run_inputs(tmp_path):
    root = tmp_path / 'd' / 'r' / 'run' / 'inputs'
    root.mkdir(parents=True)
    video = root / 'one.avi'; video.write_bytes(b'video')
    eeg = root / 'eeg.e'; eeg.write_bytes(b'eeg')
    settings = SimpleNamespace(work_root=tmp_path)
    row = SimpleNamespace(id='run', config_json={'dataset_id': 'd', 'recording_id': 'r', 'config': {'data': {'video_segments': [str(video)], 'eeg_file': str(eeg)}}}, metrics_json={'metrics': {'fusion/video_segment_manifest.json': {'sync_verified': True, 'segments': [{'filename': 'one.avi', 'sync_verified': True, 'global_start_sec': 5, 'global_end_sec': 15}]}}})
    sources = evidence_sources(settings, row)
    assert sources['videos'][0]['global_start_sec'] == 5
    assert sources['eeg_available']
    outside = tmp_path / 'other.e'; outside.write_bytes(b'private')
    with pytest.raises(ValueError): input_path(settings, row, outside)
    video.unlink()
    assert evidence_sources(settings, row)['videos'] == []


def test_evidence_api_handles_missing_inputs_and_invalid_requests(settings, migrated_db, monkeypatch):
    from fyp_data_platform import ui
    from fyp_data_platform.db.models import ModelRun
    from fyp_data_platform.storage.local import LocalStorage
    engine, factory = migrated_db
    monkeypatch.setattr(ui, 'load_settings', lambda _: settings)
    monkeypatch.setattr(ui, 'create_engine_and_session', lambda _: (engine, factory))
    monkeypatch.setattr(ui, 'create_storage', lambda _: LocalStorage(settings.local_storage_root))
    with factory.begin() as session:
        session.add(ModelRun(id='run', name='run', modality='fusion', status='completed', config_json={'dataset_id': 'd', 'recording_id': 'r'}, metrics_json={}))
    client = ui.create_app().test_client()
    assert client.get('/api/runs/missing/evidence').status_code == 404
    assert client.get('/api/runs/run/evidence').get_json() == {'videos': [], 'eeg_available': False}
    assert client.get('/api/runs/run/evidence/eeg?start=0&end=4').status_code == 400
    assert client.get('/api/runs/run/evidence/video/0/media').status_code == 404


def test_verified_video_preview_uses_retained_run_segment(settings, migrated_db, monkeypatch):
    import subprocess, time
    from fyp_data_platform import ui
    from fyp_data_platform.db.models import ModelRun
    from fyp_data_platform.storage.local import LocalStorage
    root = settings.work_root / 'd' / 'r' / 'run' / 'inputs'
    root.mkdir(parents=True)
    video = root / 'segment.avi'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=25', '-t', '1', '-c:v', 'mpeg4', str(video)], check=True)
    original = video.read_bytes()
    engine, factory = migrated_db
    monkeypatch.setattr(ui, 'load_settings', lambda _: settings)
    monkeypatch.setattr(ui, 'create_engine_and_session', lambda _: (engine, factory))
    monkeypatch.setattr(ui, 'create_storage', lambda _: LocalStorage(settings.local_storage_root))
    with factory.begin() as session:
        session.add(ModelRun(id='run', name='run', modality='fusion', status='completed', config_json={'dataset_id': 'd', 'recording_id': 'r', 'config': {'data': {'video_segments': [str(video)]}}}, metrics_json={'metrics': {'fusion/video_segment_manifest.json': {'sync_verified': True, 'segments': [{'filename': 'segment.avi', 'sync_verified': True, 'global_start_sec': 5, 'global_end_sec': 6}]}}}))
    client = ui.create_app().test_client()
    assert client.get('/api/runs/run/evidence').get_json()['videos'][0]['global_start_sec'] == 5
    for _ in range(100):
        state = client.get('/api/runs/run/evidence/video/0/preview').get_json()
        if state['status'] != 'preparing': break
        time.sleep(.1)
    assert state['status'] == 'ready'
    response = client.get(state['url'], headers={'Range': 'bytes=0-31'})
    assert response.status_code == 206
    assert b'ftyp' in response.data
    assert video.read_bytes() == original
