from io import BytesIO
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from sqlalchemy import select

from fyp_data_platform import ui
from fyp_data_platform.db.models import ModelRun
from fyp_data_platform.integrations.cancellation import ModelRunStopped, run_cancellable
from fyp_data_platform.integrations import seizure_model
from fyp_data_platform.storage.local import LocalStorage


@pytest.mark.parametrize('ignore_term', [False, True])
def test_real_process_is_reaped_on_stop(tmp_path, monkeypatch, ignore_term):
    original = subprocess.Popen
    children = []
    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    stop = threading.Event()
    timer = threading.Timer(0.5, stop.set)
    timer.start()
    try:
        with (tmp_path / 'log').open('wb') as output, pytest.raises(ModelRunStopped):
            script = 'import time, signal, os; '
            if ignore_term:
                script += "signal.signal(signal.SIGTERM, signal.SIG_IGN) if os.name == 'posix' else None; "
            run_cancellable([sys.executable, '-c', script + 'time.sleep(60)'], cwd=tmp_path, stdout=output, cancel_event=stop)
        assert len(children) == 1
        assert children[0].poll() is not None
    finally:
        timer.cancel()
        for process in children:
            if process.poll() is None:
                process.kill(); process.wait()


def test_stop_before_launch_and_normal_process_failures(tmp_path):
    stop = threading.Event(); stop.set()
    with pytest.raises(ModelRunStopped):
        run_cancellable(['must-not-launch'], cwd=tmp_path, stdout=subprocess.DEVNULL, cancel_event=stop)
    stop.clear()
    with pytest.raises(subprocess.CalledProcessError):
        run_cancellable([sys.executable, '-c', 'raise SystemExit(7)'], cwd=tmp_path, stdout=subprocess.DEVNULL, cancel_event=stop)


def test_stop_api_marks_run_stopped_releases_lock_and_keeps_source(settings, migrated_db, monkeypatch):
    engine, factory = migrated_db
    storage = LocalStorage(settings.local_storage_root)
    monkeypatch.setattr(ui, 'load_settings', lambda _: settings)
    monkeypatch.setattr(ui, 'create_engine_and_session', lambda _: (engine, factory))
    monkeypatch.setattr(ui, 'create_storage', lambda _: storage)
    monkeypatch.setattr(seizure_model, 'compute_info', lambda: {'mode': 'cpu'})
    entered = threading.Event()
    def fake_model(command, *, cwd, stdout, cancel_event):
        entered.set()
        assert cancel_event.wait(10), 'test did not send cancellation'
        raise ModelRunStopped()
    monkeypatch.setattr(seizure_model, 'run_cancellable', fake_model)
    client = ui.create_app().test_client()
    uploaded = client.post('/api/uploads', data={'mode': 'video', 'videos': (BytesIO(b'original-video'), 'sample.avi'), 'dataset_id': 'stop-test', 'patient_id': 'p1', 'recording_id': 'r1'}, content_type='multipart/form-data')
    assert uploaded.status_code == 201
    job = client.post('/api/runs', json={'dataset_id': 'stop-test', 'recording_id': 'r1', 'mode': 'video'}).get_json()
    assert entered.wait(10)
    current = client.get(f"/api/jobs/{job['id']}").get_json()
    run_id = current['run_id']
    assert client.post('/api/runs/unknown/stop').status_code == 409
    assert client.post(f'/api/runs/{run_id}/stop').status_code == 202
    for _ in range(100):
        current = client.get(f"/api/jobs/{job['id']}").get_json()
        if current['status'] == 'stopped': break
        time.sleep(0.02)
    assert current['status'] == 'stopped'
    assert client.post(f"/api/jobs/{job['id']}/stop").status_code == 409
    with factory() as session:
        run = session.get(ModelRun, run_id)
        assert run.status == 'stopped'
        assert run.completed_at is not None
        assert run.metrics_json['artifacts'] == []
    assert client.get(f'/api/runs/{run_id}/source-video').data == b'original-video'
    # A second run is accepted after cancellation; stop it too.
    entered.clear()
    second = client.post('/api/runs', json={'dataset_id': 'stop-test', 'recording_id': 'r1', 'mode': 'video'})
    assert second.status_code == 202
    assert entered.wait(10)
    assert client.post(f"/api/jobs/{second.get_json()['id']}/stop").status_code == 202
    for _ in range(100):
        if client.get(f"/api/jobs/{second.get_json()['id']}").get_json()['status'] == 'stopped': break
        time.sleep(0.02)
    assert client.get(f"/api/jobs/{second.get_json()['id']}").get_json()['status'] == 'stopped'


@pytest.mark.parametrize('stage', ['input_staging', 'artifact_persistence'])
def test_stop_before_model_creation_and_during_final_save(settings, migrated_db, monkeypatch, stage):
    engine, factory = migrated_db
    monkeypatch.setattr(ui, 'load_settings', lambda _: settings)
    monkeypatch.setattr(ui, 'create_engine_and_session', lambda _: (engine, factory))
    monkeypatch.setattr(ui, 'create_storage', lambda _: LocalStorage(settings.local_storage_root))
    entered = threading.Event(); release = threading.Event()
    class FakeIntegration:
        def __init__(self, *args): pass
        def run(self, *args, progress_callback, cancel_event, **kwargs):
            progress_callback({'stage': stage, 'run_id': 'owned-run'})
            entered.set()
            if stage == 'input_staging':
                assert cancel_event.wait(5)
                raise ModelRunStopped()
            assert release.wait(5)
            assert not cancel_event.is_set()
            return seizure_model.ModelRunResult('owned-run', 'd', 'r', 'video', 'completed', '', [], {}, [], {})
    monkeypatch.setattr(ui, 'SeizureModelIntegration', FakeIntegration)
    client = ui.create_app().test_client()
    started = client.post('/api/runs', json={'dataset_id': 'd', 'recording_id': 'r'}).get_json()
    try:
        assert entered.wait(5)
        response = client.post(f"/api/jobs/{started['id']}/stop")
        assert response.status_code == (202 if stage == 'input_staging' else 409)
    finally:
        release.set()
    for _ in range(100):
        status = client.get(f"/api/jobs/{started['id']}").get_json()['status']
        if status != 'running': break
        time.sleep(0.02)
    assert status == ('stopped' if stage == 'input_staging' else 'completed')
