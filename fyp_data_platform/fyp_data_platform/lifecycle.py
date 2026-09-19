"""Archive and delete managed data with retryable storage cleanup."""
import shutil

from sqlalchemy import select, delete
from .db.models import Recording, ModelRun, StoredFile, Prediction, Annotation, TemporalWindow
from .storage.keys import safe_entity

ACTIVE = {"created", "queued", "running"}

def run_recording_id(run):
    config = run.config_json or {}
    return f"{config.get('dataset_id')}::{config.get('recording_id')}"

def require_available(session, dataset_id, recording_id):
    recording = session.get(Recording, f"{dataset_id}::{recording_id}")
    if recording and (recording.archived or recording.deleting):
        raise ValueError("Restore the archived recording before uploading or starting a run.")

def _scope(session, kind, identity):
    if kind not in {"recordings", "runs"}:
        raise ValueError("Unknown item type")
    target = session.get(Recording if kind == "recordings" else ModelRun, identity)
    if target is None:
        raise LookupError("Item no longer exists")
    runs = [target] if kind == "runs" else [r for r in session.scalars(select(ModelRun)).all() if run_recording_id(r) == identity]
    if any(r.status in ACTIVE for r in runs):
        raise ValueError("An associated run is active. Wait until processing finishes.")
    return target, runs

def archive(session, kind, identity, value):
    target, _ = _scope(session, kind, identity)
    if target.deleting:
        raise ValueError("Deletion has started. Retry permanent deletion; this item cannot be restored.")
    if kind == "runs" and not value:
        parent = session.get(Recording, run_recording_id(target))
        if parent and (parent.archived or parent.deleting):
            raise ValueError("Restore the recording first.")
    target.archived = value

def _files(session, kind, target, runs):
    ids = {r.id for r in runs}
    return [f for f in session.scalars(select(StoredFile)).all()
            if (kind == "recordings" and f.recording_id == target.id)
            or (f.role == "model-output" and (f.metadata_json or {}).get("run_id") in ids)]

def preview(session, kind, identity):
    target, runs = _scope(session, kind, identity)
    files = _files(session, kind, target, runs)
    return {"id": identity, "runs": len(runs), "files": len(files),
            "bytes": sum(f.size_bytes or 0 for f in files), "archived": target.archived,
            "deleting": target.deleting,
            "confirmation": identity,
            "message": "Removes managed files, results and working copies. Original files on your computer are kept."}

def _work_path(settings, kind, target):
    config = target.config_json if kind == "runs" else {"dataset_id": target.dataset_id, "recording_id": target.source_recording_id}
    dataset, recording = config.get("dataset_id"), config.get("recording_id")
    if not dataset or not recording:
        raise ValueError("Cannot safely identify the managed working directory")
    # Sanitized aliases could collide with other recordings. Never delete through an alias.
    if safe_entity(dataset) != dataset or safe_entity(recording) != recording:
        raise ValueError("Unsafe working-directory identity")
    root = settings.work_root.resolve()
    path = root / dataset / recording
    if kind == "runs":
        if safe_entity(target.id) != target.id:
            raise ValueError("Unsafe run identity")
        path /= target.id
    if path.is_symlink() or path.resolve() != path.absolute() or root not in path.resolve().parents:
        raise ValueError("Refusing cleanup outside the managed working directory")
    return path

def permanently_delete(factory, storage, settings, kind, identity, confirmation):
    if confirmation != identity:
        raise ValueError("Type the exact item ID to confirm permanent deletion")
    # Commit a tombstone before irreversible I/O. Failure keeps the archived item
    # visible in Settings for a retry, and restoration is blocked.
    with factory.begin() as session:
        target, runs = _scope(session, kind, identity)
        if not target.archived:
            raise ValueError("Archive this item before deleting it")
        files = _files(session, kind, target, runs)
        file_ids = [f.id for f in files]
        run_ids = [r.id for r in runs]
        path = _work_path(settings, kind, target)
        # Local Windows paths are case-insensitive. Reject aliases which would
        # clean another catalog entry's working copies.
        if kind == "recordings":
            for other in session.scalars(select(Recording)).all():
                if other.id == identity:
                    continue
                try:
                    other_path = _work_path(settings, "recordings", other)
                except ValueError:
                    continue
                if other_path == path:
                    raise ValueError("Working directory is shared by another recording")
        keys = []
        for f in files:
            if f.storage_backend != settings.storage_backend:
                raise ValueError("File uses a different storage backend; deletion cannot safely continue")
            if f.payload_state == "stored":
                keys.append(f.storage_key)
        # Do not break another recording's annotations/features or another run.
        for a in session.scalars(select(Annotation).where(Annotation.source_file_id.in_(file_ids))).all():
            if kind != "recordings" or a.recording_id != identity:
                raise ValueError("A file is referenced by another annotation")
        for w in session.scalars(select(TemporalWindow).where(TemporalWindow.feature_file_id.in_(file_ids))).all():
            if kind != "recordings" or w.recording_id != identity:
                raise ValueError("A file is referenced by another recording's features")
        for r in session.scalars(select(ModelRun).where(ModelRun.artifact_file_id.in_(file_ids))).all():
            if r.id not in run_ids:
                raise ValueError("A file is referenced by another model run")
        target.deleting = True
        for r in runs:
            r.deleting = True
            r.archived = True
    for key in set(keys):
        storage.delete(key)
    if path.exists():
        # Every descendant is checked before traversal; junctions are rejected too.
        for child in path.rglob('*'):
            if child.is_symlink() or (hasattr(child, 'is_junction') and child.is_junction()):
                raise ValueError("Linked files found in working directory; cleanup stopped")
        shutil.rmtree(path)
    with factory.begin() as session:
        session.execute(delete(Prediction).where(Prediction.model_run_id.in_(run_ids)))
        if kind == "recordings":
            windows = select(TemporalWindow.window_id).where(TemporalWindow.recording_id == identity)
            session.execute(delete(Prediction).where(Prediction.window_id.in_(windows)))
            session.execute(delete(TemporalWindow).where(TemporalWindow.recording_id == identity))
            session.execute(delete(Annotation).where(Annotation.recording_id == identity))
        session.execute(delete(ModelRun).where(ModelRun.id.in_(run_ids)))
        session.execute(delete(StoredFile).where(StoredFile.id.in_(file_ids)))
        if kind == "recordings":
            session.execute(delete(Recording).where(Recording.id == identity))
    return {"deleted": identity, "files": len(file_ids), "runs": len(run_ids)}
