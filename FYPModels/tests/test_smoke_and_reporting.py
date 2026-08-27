from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from seizure_multimodal.evaluation.report import comparison_rows
from seizure_multimodal.reproducibility import build_run_record
from seizure_multimodal.smoke import BANNER, run_synthetic_smoke_test


class SmokeAndReportingTests(unittest.TestCase):
    def test_synthetic_smoke_is_clearly_nonclinical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = run_synthetic_smoke_test(directory)
            report = json.loads((Path(directory) / "report.json").read_text(encoding="utf-8"))
            self.assertTrue((Path(directory) / "synthetic_manifest.csv").exists())
            self.assertTrue((Path(directory) / "run_metadata.json").exists())
        self.assertEqual(payload["warning"], BANNER)
        self.assertFalse(report["paired_real_veeg_verified"])
        self.assertIn("not performed", report["real_multimodal_notice"])

    def test_real_fusion_metrics_are_gated(self) -> None:
        rows = comparison_rows(
            {"Late Fusion": {"F1": 0.99}, "Feature Fusion": {"AUROC": 1.0}},
            paired_real_veeg_verified=False,
        )
        late = next(row for row in rows if row["Model"] == "Late Fusion")
        feature = next(row for row in rows if row["Model"] == "Feature Fusion")
        self.assertEqual(late["F1"], "N/A")
        self.assertEqual(feature["AUROC"], "N/A")

    def test_run_record_requires_reproducibility_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing fields"):
            build_run_record({"experiment_id": "incomplete"})


if __name__ == "__main__":
    unittest.main()
