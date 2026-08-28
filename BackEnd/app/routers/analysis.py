from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_store
from ..models import Metrics, ScoredResponse, ScoringRequest, TestStatus
from ..services.results_store import ResultsStore
from ..services.scoring import Annotation, score

router = APIRouter(prefix="/api", tags=["analysis"])


@router.post("/recordings/{recording_id}/score", response_model=ScoredResponse)
def rescore(recording_id: str, req: ScoringRequest,
            store: ResultsStore = Depends(get_store)):
    """Recompute metrics at a chosen threshold and persistence.

    The frontend calls this on every slider move rather than doing the maths
    itself, so what a clinician sees is always what the backend would report.
    """
    try:
        df = store.timeline(recording_id)
        rec = store.recording(recording_id)
    except FileNotFoundError:
        raise HTTPException(404, f"no results for {recording_id}")

    result = score(
        df.start_sec.values, df.video_probability.values,
        Annotation(rec["eeg_onset_sec"], rec["clinical_onset_sec"],
                   rec["seizure_offset_sec"]),
        threshold=req.threshold, min_consecutive=req.min_consecutive,
        window_sec=rec["window_sec"],
    )
    pred = result.pop("prediction")
    return {"recording_id": recording_id, "threshold": req.threshold,
            "min_consecutive": req.min_consecutive,
            "metrics": Metrics(**result), "prediction": pred}


@router.get("/test-matrix", response_model=list[TestStatus])
def test_matrix(store: ResultsStore = Depends(get_store)):
    """What actually ran, and why the rest did not.

    Surfaced as a first-class endpoint because a dashboard that shows only the
    completed test implies the others succeeded.
    """
    rows = store.summary()
    if not rows:
        return []
    out = []
    for r in rows:
        yes = lambda k: str(r.get(k, "")).strip().lower() in {"yes", "true", "1"}
        out.append({
            "test": str(r.get("Test", "")),
            "status": str(r.get("Status", "skipped")).strip().lower(),
            "video_used": yes("Video"), "eeg_used": yes("EEG"),
            "fusion_used": yes("Fusion"),
            "reason": None if str(r.get("Pairing", "")) in ("N/A", "nan")
                      else str(r.get("Pairing")),
        })
    return out
