"""Regression test: the API must reproduce the recorded run exactly.

If a refactor changes any of these numbers, the dashboard has started
disagreeing with results/video_only/Sz2PG_metrics.json and the test fails.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.scoring import Annotation, apply_persistence, count_alarm_events, score

import numpy as np

CSV = Path(__file__).parent / "Sz2PG_probability_timeline.csv"
ANN = Annotation(eeg_onset_sec=624.0, clinical_onset_sec=640.0, seizure_offset_sec=None)

# Values copied from the recorded metrics.json, not from this code.
RECORDED = dict(sensitivity=0.7397959183673469, specificity=0.5040128410914928,
                auroc=0.7377649294067546, f1=0.4461538461538461,
                precision=0.31938325991189426, cohen_kappa=0.1680164342432742,
                fp_per_hour=351.9230769230769, tn=314, fp=309, fn=51, tp=145,
                alarm_events=61, onset_error=-8.369000000000028)


@pytest.fixture(scope="module")
def timeline():
    d = pd.read_csv(CSV)
    return d.start_sec.values, d.video_probability.values


def test_reproduces_recorded_metrics(timeline):
    start, prob = timeline
    r = score(start, prob, ANN, threshold=0.5, min_consecutive=1)
    for key in ("sensitivity", "specificity", "auroc", "f1", "precision", "cohen_kappa"):
        assert r[key] == pytest.approx(RECORDED[key], abs=2e-3), key
    assert r["alarm_events"] == RECORDED["alarm_events"]
    assert r["false_positives_per_hour"] == pytest.approx(RECORDED["fp_per_hour"], rel=1e-3)
    assert r["onset_error_sec"] == pytest.approx(RECORDED["onset_error"], abs=1e-2)
    for k in ("fp", "fn", "tp"):
        assert r[k] == RECORDED[k], k


def test_detects_before_clinical_onset(timeline):
    """The headline finding: the alarm precedes the clinician's mark."""
    start, prob = timeline
    r = score(start, prob, ANN)
    assert r["onset_error_sec"] < 0
    assert r["first_alarm_sec"] == pytest.approx(631.631, abs=1e-2)


def test_persistence_suppresses_false_alarms(timeline):
    start, prob = timeline
    loose = score(start, prob, ANN, threshold=0.5, min_consecutive=1)
    tight = score(start, prob, ANN, threshold=0.99, min_consecutive=5)
    assert tight["specificity"] > loose["specificity"]
    assert tight["false_positives_per_hour"] < loose["false_positives_per_hour"] / 10
    assert tight["onset_error_sec"] < 0   # still early


def test_transition_window_excluded(timeline):
    """Windows between EEG and clinical onset must not be scored."""
    start, prob = timeline
    r = score(start, prob, ANN)
    n_transition = int(((start >= 624.0) & (start < 640.0)).sum())
    assert n_transition > 0
    assert r["tn"] + r["fp"] + r["fn"] + r["tp"] == len(start) - n_transition


def test_offset_never_invented(timeline):
    start, prob = timeline
    assert score(start, prob, ANN)["offset_error_sec"] is None


def test_persistence_logic():
    a = np.array([1, 0, 1, 1, 1, 0, 1, 1], dtype=bool)
    assert apply_persistence(a, 1).tolist() == [1, 0, 1, 1, 1, 0, 1, 1]
    assert apply_persistence(a, 3).tolist() == [0, 0, 1, 1, 1, 0, 0, 0]
    assert apply_persistence(a, 4).tolist() == [0] * 8


def test_alarm_events_count_runs_not_windows():
    assert count_alarm_events(np.array([1, 1, 1, 0, 0, 1, 1])) == 2
    assert count_alarm_events(np.zeros(5, dtype=int)) == 0


def test_missing_annotation_scores_nothing():
    r = score([0, 1, 2], [0.9, 0.9, 0.9],
              Annotation(eeg_onset_sec=None, clinical_onset_sec=None))
    assert r["tn"] == r["fp"] == r["fn"] == r["tp"] == 0
    assert r["auroc"] is None
