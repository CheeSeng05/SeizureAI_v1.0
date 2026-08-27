from __future__ import annotations

from typing import Any, Mapping

from ..config import validate_training_policy


def _set_trainability(component: Any, *, freeze: bool) -> None:
    if not hasattr(component, "parameters"):
        raise TypeError("Trainable components must expose a PyTorch-style parameters() method")
    for parameter in component.parameters():
        parameter.requires_grad = not freeze
    if freeze and hasattr(component, "eval"):
        component.eval()


def prepare_trainable_components(
    *,
    config: Mapping[str, Any],
    paired_data_verified: bool,
    video_encoder: Any,
    eeg_encoder: Any,
    fusion_model: Any,
) -> list[dict[str, Any]]:
    """Apply explicit future-training policy and return optimizer parameter groups.

    No optimizer step or training loop is run here. Calling this function with the
    default inference config stops before mutating model trainability.
    """

    validate_training_policy(config, paired_data_verified=paired_data_verified)
    video_cfg = config.get("video_encoder", {})
    eeg_cfg = config.get("eeg_encoder", {})
    video_freeze = bool(video_cfg.get("freeze", True))
    eeg_freeze = bool(eeg_cfg.get("freeze", True))
    _set_trainability(video_encoder, freeze=video_freeze)
    _set_trainability(eeg_encoder, freeze=eeg_freeze)
    _set_trainability(fusion_model, freeze=False)
    groups: list[dict[str, Any]] = [
        {
            "name": "fusion",
            "params": [p for p in fusion_model.parameters() if p.requires_grad],
            "lr": float(config.get("fusion", {}).get("learning_rate", 1e-3)),
        }
    ]
    if not video_freeze:
        groups.append(
            {
                "name": "video_encoder",
                "params": [p for p in video_encoder.parameters() if p.requires_grad],
                "lr": float(video_cfg.get("learning_rate", 1e-5)),
            }
        )
    if not eeg_freeze:
        groups.append(
            {
                "name": "eeg_encoder",
                "params": [p for p in eeg_encoder.parameters() if p.requires_grad],
                "lr": float(eeg_cfg.get("learning_rate", 1e-5)),
            }
        )
    return groups
