"""SeizureAI API.

Serves what the model actually produced. It computes no predictions of its own -
inference happens in the model repo via `python -m seizure_multimodal test`, and
this reads the results it writes.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .dependencies import get_store
from .routers import analysis, recordings, uploads

app = FastAPI(
    title="SeizureAI API",
    description="Read-only access to VSViG inference results, plus live "
                "threshold scoring.",
    version="0.1.0",
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"], allow_headers=["*"],
)

app.include_router(recordings.router)
app.include_router(analysis.router)
app.include_router(uploads.router)


@app.get("/api/health")
def health():
    store = get_store()
    ids = store.recording_ids()
    return {
        "status": "ok" if ids else "no results found",
        "results_dir": str(settings.results_dir.resolve()),
        "recordings": len(ids),
        "recording_ids": ids,
    }