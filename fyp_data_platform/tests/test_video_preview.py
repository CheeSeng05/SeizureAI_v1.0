from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
import shutil
import subprocess
import time

import pytest

from fyp_data_platform import ui
from fyp_data_platform.db.models import ModelRun
from fyp_data_platform.db.repository import CatalogRepository
from fyp_data_platform.storage.local import LocalStorage
from fyp_data_platform.video_preview import VideoPreviews


def test_elapsed_time_available_before_final_metrics():
    start = datetime.now(timezone.utc) - timedelta(seconds=65)
    row = ModelRun(status='running', started_at=start, config_json={}, metrics_json={})
    assert 64 <= ui._run_summary(row)['elapsed_sec'] <= 68
    row.status = 'failed'
    row.completed_at = start + timedelta(seconds=70)
    assert ui._run_summary(row)['elapsed_sec'] == 70
    row.started_at = None
    assert ui._run_summary(row)['elapsed_sec'] is None


def test_preview_deduplicates_work_and_missing_encoder_is_clear(tmp_path, monkeypatch):
    manager = VideoPreviews(tmp_path, None)
    path = manager.path('dataset', 'recording', 'source.avi', 'digest')
    monkeypatch.setattr(shutil, 'which', lambda _: None)
    assert manager.ensure(path, 'source.avi')['status'] == 'failed'
    monkeypatch.setattr(shutil, 'which', lambda _: '/usr/bin/ffmpeg')
    submitted = []
    monkeypatch.setattr(manager.executor, 'submit', lambda *args: submitted.append(args))
    assert manager.ensure(path, 'source.avi')['status'] == 'preparing'
    assert manager.ensure(path, 'source.avi')['status'] == 'preparing'
    assert len(submitted) == 1
    assert manager.busy
    assert manager.path('dataset', 'recording', 'source.avi', 'other') != path
    manager.executor.shutdown()


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg is installed in the application image')
def test_avi_preview_conversion_ranges_and_original_preserved(settings, migrated_db, monkeypatch, tmp_path):
    original = tmp_path / 'sample.avi'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=25',
                    '-t', '2', '-c:v', 'mpeg4', str(original)], check=True)
    original_bytes = original.read_bytes()
    engine, factory = migrated_db
    storage = LocalStorage(settings.local_storage_root)
    monkeypatch.setattr(ui, 'load_settings', lambda _: settings)
    monkeypatch.setattr(ui, 'create_engine_and_session', lambda _: (engine, factory))
    monkeypatch.setattr(ui, 'create_storage', lambda _: storage)
    client = ui.create_app().test_client()
    response = client.post('/api/uploads', data={
        'mode': 'video', 'videos': (BytesIO(original_bytes), 'sample.avi'),
        'dataset_id': 'ui-recordings', 'patient_id': 'p1', 'recording_id': 'avi-1',
    }, content_type='multipart/form-data')
    assert response.status_code == 201
    with factory.begin() as session:
        CatalogRepository(session).model_run(
            'preview-run', name='video', model_type='video', model_version=None,
            modality='video', status='running', git_commit=None,
            config_json={'dataset_id': 'ui-recordings', 'recording_id': 'avi-1'},
            metrics_json={}, artifact_file_id=None,
            started_at=datetime.now(timezone.utc), completed_at=None,
        )
    progress_path = settings.work_root / 'ui-recordings/avi-1/preview-run/progress.json'
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(json.dumps({'completed_units': 1, 'total_units': 10}))
    detail = client.get('/api/runs/preview-run').get_json()
    assert detail['elapsed_sec'] >= 0
    assert detail['progress']['completed_units'] == 1
    assert detail['source_video']['preview_status_url'] == '/api/runs/preview-run/preview'
    assert client.get('/api/runs/missing/preview').status_code == 404
    assert client.get('/api/runs/preview-run/preview-video').status_code == 409
    for _ in range(100):
        state = client.get('/api/runs/preview-run/preview').get_json()
        if state['status'] != 'preparing':
            break
        time.sleep(0.1)
    assert state['status'] == 'ready', state
    response = client.get(state['url'], headers={'Range': 'bytes=0-31'})
    assert response.status_code == 206
    assert response.mimetype == 'video/mp4'
    assert b'ftyp' in response.data
    assert len(response.data) == 32
    assert response.headers['Content-Range'].startswith('bytes 0-31/')
    assert client.get(state['url'], headers={'Range': 'bytes=-16'}).status_code == 206
    assert client.get(state['url'], headers={'Range': 'bytes=999999999-'}).status_code == 416
    assert client.get('/api/runs/preview-run/source-video').data == original_bytes
    cached = next(settings.work_root.rglob('browser-preview/*.mp4'))
    metadata = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(cached)]))['streams'][0]
    assert metadata['codec_name'] == 'h264'
    assert metadata['pix_fmt'] == 'yuv420p'
    assert float(metadata['duration']) == pytest.approx(2, abs=0.04)
    assert int(metadata['nb_frames']) == 50


def test_failed_conversion_never_publishes_partial_preview(tmp_path):
    storage = LocalStorage(tmp_path / 'storage')
    manager = VideoPreviews(tmp_path / 'work', storage)
    output = manager.path('d', 'r', 'missing.avi', 'hash')
    manager._convert(output, 'missing.avi')
    assert not output.exists()
    assert manager.jobs[output]['status'] == 'failed'
    manager.executor.shutdown()
