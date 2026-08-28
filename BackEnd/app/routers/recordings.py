from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_store
from ..models import Recording, RunProvenance, TestStatus, Timeline
from ..services.results_store import ResultsStore

router = APIRouter(prefix="/api/recordings", tags=["recordings"])


@router.get("", response_model=list[Recording])
def list_recordings(store: ResultsStore = Depends(get_store)):
    return [store.recording(r) for r in store.recording_ids()]


@router.get("/{recording_id}", response_model=Recording)
def get_recording(recording_id: str, store: ResultsStore = Depends(get_store)):
    try:
        return store.recording(recording_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no results for {recording_id}")


@router.get("/{recording_id}/timeline", response_model=Timeline)
def get_timeline(recording_id: str, store: ResultsStore = Depends(get_store)):
    try:
        df = store.timeline(recording_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no results for {recording_id}")
    rec = store.recording(recording_id)
    return {
        "recording_id": recording_id,
        "window_sec": rec["window_sec"],
        "start_sec": df.start_sec.round(3).tolist(),
        "probability": df.video_probability.round(6).tolist(),
        "prediction": df.video_prediction.astype(int).tolist(),
    }


@router.get("/{recording_id}/provenance", response_model=RunProvenance)
def get_provenance(recording_id: str, store: ResultsStore = Depends(get_store)):
    try:
        return store.provenance(recording_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no results for {recording_id}")
