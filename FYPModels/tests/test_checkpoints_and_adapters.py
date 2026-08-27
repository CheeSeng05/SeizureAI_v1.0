from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from seizure_multimodal.checkpoints.inspection import (
    inspect_checkpoint_archive,
    read_state_dict_metadata,
)
from seizure_multimodal.checkpoints.provenance import sha256_file
from seizure_multimodal.eeg.eeg_adapter import EEGAdapter


ROOT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
    def test_actual_vsvig_checkpoint_structure_and_embedding(self) -> None:
        path = ROOT / "VSViG-main" / "VSViG-base.pth"
        archive = inspect_checkpoint_archive(path)
        self.assertTrue(archive["is_zip_archive"])
        state = read_state_dict_metadata(path)
        self.assertEqual(len(state), 1283)
        self.assertEqual(state["stem.stem.0.weight"].shape, (24, 3, 32, 32))
        self.assertEqual(state["stem_pe.stem.0.weight"].shape, (24, 3, 1, 1))
        self.assertEqual(state["fc.0.weight"].shape, (256, 384, 1, 1))
        self.assertEqual(state["fc.3.weight"].shape, (1, 256, 1, 1))

    def test_original_checkpoint_hash_is_stable(self) -> None:
        self.assertEqual(
            sha256_file(ROOT / "VSViG-main" / "VSViG-base.pth"),
            "0a95e3e0093730dc535e21a84f04943c73549a2b6a19a85e33aebc7c64f965a9",
        )

    def test_pose_checkpoint_structure(self) -> None:
        state = read_state_dict_metadata(ROOT / "VSViG-main" / "pose.pth")
        self.assertEqual(len(state), 250)
        self.assertEqual(state["model.0.0.weight"].shape, (32, 3, 3, 3))
        self.assertEqual(state["refinement_stages.0.heatmaps.1.0.weight"].shape, (19, 128, 1, 1))
        self.assertEqual(state["refinement_stages.0.pafs.1.0.weight"].shape, (38, 128, 1, 1))

    def test_missing_eeg_checkpoint_never_creates_random_weights(self) -> None:
        missing = ROOT / "checkpoints" / "pretrained" / "eeg" / "missing.h5"
        with self.assertRaisesRegex(FileNotFoundError, "Do not initialize random replacement"):
            EEGAdapter(checkpoint_path=missing)
        self.assertFalse(missing.exists())

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("timm"), "PyTorch/timm unavailable")
    def test_vsvig_strict_checkpoint_load_and_inference_mode(self) -> None:
        from seizure_multimodal.video.vsvig_adapter import VSViGAdapter

        adapter = VSViGAdapter(
            source_path=ROOT / "VSViG-main" / "VSViG.py",
            checkpoint_path=ROOT / "VSViG-main" / "VSViG-base.pth",
            dynamic_partition_path=ROOT / "VSViG-main" / "dy_point_order.pt",
        )
        try:
            self.assertTrue(adapter.is_inference_only)
            self.assertEqual(adapter.embedding_dim, 384)
        finally:
            adapter.close()

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("cv2"),
        "PyTorch/OpenCV unavailable",
    )
    def test_pose_checkpoint_strict_load_and_inference_mode(self) -> None:
        from seizure_multimodal.video.raw_pipeline import LightweightOpenPose

        adapter = LightweightOpenPose(
            source_directory=ROOT / "third_party" / "lightweight_openpose",
            checkpoint_path=ROOT / "VSViG-main" / "pose.pth",
            device="cpu",
        )
        self.assertTrue(adapter.is_inference_only)


if __name__ == "__main__":
    unittest.main()
