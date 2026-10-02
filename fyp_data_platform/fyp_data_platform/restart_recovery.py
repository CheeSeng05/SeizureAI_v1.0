"""Reconcile interrupted runs before the Docker web server starts.

Only use with the single app container and its private work volume. Never run
recovery in a request handler: a live worker may still own a run.
"""
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from fyp_data_platform.db.models import ModelRun
from fyp_data_platform.storage.keys import safe_entity


def recover_interrupted_runs(session_factory, work_root):
    recovered = []
    with session_factory.begin() as session:
        rows = session.scalars(select(ModelRun).where(
            ModelRun.status == 'running', ModelRun.model_type == 'seizure_multimodal'
        ))
        for row in rows:
            config = row.config_json or {}
            if not config.get('dataset_id') or not config.get('recording_id'):
                continue
            run_root = Path(work_root) / safe_entity(config['dataset_id']) / safe_entity(config['recording_id']) / safe_entity(row.id)
            if not (run_root / 'inference.yaml').is_file():
                continue
            row.status = 'stopped'
            # The actual termination time is unknown after a forced shutdown.
            # Do not count downtime as model processing time.
            row.metrics_json = {
                **(row.metrics_json or {}),
                'error': 'Run interrupted when the app stopped or restarted. Start a new run to process this recording.',
                'recovered_at': datetime.now(timezone.utc).isoformat(),
            }
            recovered.append(row.id)
    return recovered


def main():
    # Fail closed if someone invokes this against an already-running container.
    proc = Path('/proc')
    if not proc.is_dir():
        raise RuntimeError('Restart recovery requires the Docker Linux process namespace.')
    for path in proc.glob('[0-9]*/cmdline'):
        try:
            args = path.read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        if b'seizure_multimodal' in args or any(arg.endswith(b'/gunicorn') for arg in args):
            raise RuntimeError('Cannot recover runs while the app or model is active.')
    from fyp_data_platform.config.settings import load_settings
    from fyp_data_platform.db.session import create_engine_and_session
    settings = load_settings()
    engine, factory = create_engine_and_session(settings.database_url)
    try:
        for run_id in recover_interrupted_runs(factory, settings.work_root):
            print(f'Marked interrupted model run {run_id} as stopped', flush=True)
    finally:
        engine.dispose()


if __name__ == '__main__':
    main()
