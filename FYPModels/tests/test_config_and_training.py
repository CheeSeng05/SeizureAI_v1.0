from __future__ import annotations

import unittest
from pathlib import Path

from seizure_multimodal.config import (
    TRAINING_DISABLED_MESSAGE,
    TrainingDisabledError,
    assert_training_enabled,
    load_config,
    validate_training_policy,
)


ROOT = Path(__file__).resolve().parents[1]


class TrainingSafeguardTests(unittest.TestCase):
    def test_default_inference_config_disables_training(self) -> None:
        config = load_config(ROOT / "configs" / "inference.yaml")
        with self.assertRaises(TrainingDisabledError) as raised:
            assert_training_enabled(config)
        self.assertEqual(str(raised.exception), TRAINING_DISABLED_MESSAGE)

    def test_future_config_is_also_disabled_at_rest(self) -> None:
        config = load_config(ROOT / "configs" / "training_future.yaml")
        with self.assertRaises(TrainingDisabledError):
            validate_training_policy(config, paired_data_verified=True)

    def test_training_requires_paired_data_even_when_enabled(self) -> None:
        config = {"training": {"enabled": True}, "fusion": {"trainable": True}}
        with self.assertRaisesRegex(ValueError, "verified synchronized"):
            validate_training_policy(config, paired_data_verified=False)

    def test_explicit_future_training_policy_can_pass(self) -> None:
        config = {"training": {"enabled": True}, "fusion": {"trainable": True}}
        validate_training_policy(config, paired_data_verified=True)


if __name__ == "__main__":
    unittest.main()
