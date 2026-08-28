"""Pydantic schemas — the contract between backend and frontend.

Field names mirror the model's own output vocabulary (window, eeg_onset_sec,
clinical_onset_sec) so there is no translation layer to get out of sync.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Recording(BaseModel):
    recording_id: str
    patient_id: str
    seizure_id: str
    seizure_type: str = Field(description="P = focal, PG = partial-to-generalised")
    source_file: str
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int
    eeg_onset_sec: Optional[float]
    clinical_onset_sec: Optional[float]
    seizure_offset_sec: Optional[float]
    alignment_status: str
    window_count: int
    window_sec: float


class TestStatus(BaseModel):
    """One row of the test matrix. Mirrors results/summary.csv."""
    test: str
    status: Literal["completed", "blocked", "skipped"]
    video_used: bool
    eeg_used: bool
    fusion_used: bool
    reason: Optional[str] = None


class RunProvenance(BaseModel):
    test_mode: str
    video_model_used: bool
    eeg_model_used: bool
    fusion_used: bool
    pairing_type: str
    training_performed: bool
    video_checkpoint: Optional[str]
    eeg_checkpoint: Optional[str]
    label_policy: Optional[str] = None


class Metrics(BaseModel):
    sensitivity: Optional[float]
    specificity: Optional[float]
    precision: Optional[float]
    auroc: Optional[float]
    f1: Optional[float]
    cohen_kappa: Optional[float]
    false_positives_per_hour: Optional[float]
    alarm_events: Optional[int]
    tn: int
    fp: int
    fn: int
    tp: int
    onset_error_sec: Optional[float] = None
    offset_error_sec: Optional[float] = None
    first_alarm_sec: Optional[float] = None


class Timeline(BaseModel):
    recording_id: str
    window_sec: float
    start_sec: list[float]
    probability: list[float]
    prediction: list[int]


class ScoringRequest(BaseModel):
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_consecutive: int = Field(1, ge=1, le=60,
                                 description="windows that must exceed threshold "
                                             "before an alarm is raised")


class ScoredResponse(BaseModel):
    recording_id: str
    threshold: float
    min_consecutive: int
    metrics: Metrics
    prediction: list[int]