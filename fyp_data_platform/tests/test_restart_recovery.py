from fyp_data_platform.db.models import ModelRun
from fyp_data_platform.restart_recovery import recover_interrupted_runs


def test_restart_recovers_only_local_interrupted_model_runs(settings, migrated_db):
    _, factory = migrated_db
    for identity, status, local in [('interrupted', 'running', True), ('finished', 'completed', True), ('elsewhere', 'running', False)]:
        with factory.begin() as session:
            session.add(ModelRun(id=identity, name=identity, model_type='seizure_multimodal', status=status,
                config_json={'dataset_id': 'd', 'recording_id': 'r'}, metrics_json={'artifacts': []}))
        if local:
            root = settings.work_root / 'd' / 'r' / identity
            root.mkdir(parents=True)
            (root / 'inference.yaml').write_text('testing: {}')
    assert recover_interrupted_runs(factory, settings.work_root) == ['interrupted']
    assert recover_interrupted_runs(factory, settings.work_root) == []
    with factory() as session:
        interrupted = session.get(ModelRun, 'interrupted')
        assert interrupted.status == 'stopped'
        assert interrupted.completed_at is None
        assert 'interrupted' in interrupted.metrics_json['error']
        assert interrupted.metrics_json['artifacts'] == []
        assert session.get(ModelRun, 'finished').status == 'completed'
        assert session.get(ModelRun, 'elsewhere').status == 'running'
    assert (settings.work_root / 'd' / 'r' / 'interrupted' / 'inference.yaml').is_file()
