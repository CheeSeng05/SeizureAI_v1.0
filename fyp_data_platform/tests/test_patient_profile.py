from io import BytesIO
from fyp_data_platform import ui
from fyp_data_platform.db.models import ModelRun
from fyp_data_platform.storage.local import LocalStorage


def test_patient_profile_saved_shared_validated_and_preserved(settings, migrated_db, monkeypatch):
    engine, factory = migrated_db
    monkeypatch.setattr(ui, 'load_settings', lambda _: settings)
    monkeypatch.setattr(ui, 'create_engine_and_session', lambda _: (engine, factory))
    monkeypatch.setattr(ui, 'create_storage', lambda _: LocalStorage(settings.local_storage_root))
    client = ui.create_app().test_client()
    def upload(recording, patient='p'):
        return client.post('/api/uploads', data={'mode': 'video', 'videos': (BytesIO(b'video'), 'sample.avi'), 'dataset_id': 'd', 'patient_id': patient, 'recording_id': recording}, content_type='multipart/form-data')
    assert upload('r').status_code == 201
    with factory.begin() as session:
        for identity in ['run', 'second']:
            session.add(ModelRun(id=identity, name=identity, status='completed', config_json={'dataset_id': 'd', 'recording_id': 'r'}, metrics_json={}))
    assert client.get('/api/runs/run').get_json()['patient']['age'] is None
    result = client.patch('/api/runs/run/patient', json={'age': 24, 'notes': 'Review history'})
    assert result.status_code == 200
    assert client.get('/api/runs/second').get_json()['patient']['notes'] == 'Review history'
    for age in [-1, 131, True, 2.5, '24']:
        assert client.patch('/api/runs/run/patient', json={'age': age}).status_code == 400
    assert client.patch('/api/runs/run/patient', json={'notes': 'x' * 4001}).status_code == 400
    assert client.patch('/api/runs/missing/patient', json={}).status_code == 404
    assert upload('new-recording').status_code == 201
    assert client.get('/api/runs/run').get_json()['patient']['age'] == 24
    assert client.patch('/api/runs/run/patient', json={'age': None, 'notes': ''}).status_code == 200
    assert client.get('/api/runs/second').get_json()['patient']['age'] is None
