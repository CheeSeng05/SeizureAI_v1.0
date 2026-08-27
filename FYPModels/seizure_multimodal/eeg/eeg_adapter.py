from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..types import EncoderOutput


@dataclass(frozen=True)
class EEGPreprocessor:
    """Preprocessing reproduced from the local Kaggle-derived notebook."""

    sampling_rate: float = 256.0
    low_freq: float = 0.5
    high_freq: float = 100.0
    notch_freq: float = 60.0
    filter_order: int = 4
    clip_sigma: float = 5.0

    def transform(self, channel_major: Any) -> np.ndarray:
        try:
            from scipy import signal
            from scipy.signal import butter, filtfilt
        except ImportError as exc:
            raise RuntimeError("SciPy is required for EEG preprocessing") from exc
        data = np.asarray(channel_major, dtype=np.float64)
        if data.ndim not in (2, 3):
            raise ValueError("EEG preprocessing expects [channels, samples] or [batch, channels, samples]")
        squeeze = data.ndim == 2
        if squeeze:
            data = data[None, ...]
        nyquist = self.sampling_rate / 2.0
        if not 0 < self.low_freq < min(self.high_freq, nyquist):
            raise ValueError("Invalid EEG bandpass frequencies")
        band_b, band_a = butter(
            self.filter_order,
            [self.low_freq / nyquist, min(self.high_freq / nyquist, 0.99)],
            btype="band",
        )
        notch_b, notch_a = signal.iirnotch(self.notch_freq / nyquist, 30)
        filtered = np.empty_like(data)
        for batch in range(data.shape[0]):
            for channel in range(data.shape[1]):
                values = filtfilt(band_b, band_a, data[batch, channel])
                values = filtfilt(notch_b, notch_a, values)
                values = (values - values.mean()) / (values.std() + 1e-8)
                filtered[batch, channel] = np.clip(values, -self.clip_sigma, self.clip_sigma)
        result = filtered.astype(np.float32)
        return result[0] if squeeze else result


def build_reference_eeg_model(input_shape: tuple[int, int] = (1024, 18)) -> Any:
    """Rebuild the exact topology defined in notebook cell 13, without training."""

    try:
        from tensorflow.keras import Model, layers, regularizers
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to build the EEG model") from exc
    inputs = layers.Input(shape=input_shape, name="eeg_input")
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(128, 7, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True, dropout=0.2))(x)
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
    x = layers.LayerNormalization()(attention + x)
    x = layers.Concatenate()(
        [layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)]
    )
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.01),
        name="eeg_dense_embedding",
    )(x)
    embedding = layers.Dropout(0.3, name="eeg_embedding")(x)
    output = layers.Dense(1, activation="sigmoid", name="seizure_output")(embedding)
    return Model(inputs, output, name="SeizureDetector")


class EEGAdapter:
    """Inference-only adapter for the local notebook's Keras checkpoint."""

    sampling_rate = 256.0
    window_seconds = 4.0
    window_samples = 1024
    channels = 18
    embedding_dim = 32

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        preprocessor: EEGPreprocessor | None = None,
        channel_order: Sequence[str] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"EEG checkpoint not found: {self.checkpoint_path}. The local notebook references "
                "best_seizure_model.h5/seizure_detector_model.h5 but neither artifact is present. "
                "Do not initialize random replacement weights."
            )
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required for EEG inference") from exc
        self.tf = tf
        self.preprocessor = preprocessor or EEGPreprocessor()
        self.channel_order = tuple(channel_order or ())
        if self.channel_order and len(self.channel_order) != self.channels:
            raise ValueError("EEG channel_order must contain exactly 18 channels")
        try:
            self.model = tf.keras.models.load_model(self.checkpoint_path, compile=False)
        except Exception:
            if not self.checkpoint_path.name.endswith(".weights.h5"):
                raise
            self.model = build_reference_eeg_model()
            self.model.load_weights(self.checkpoint_path)
        self.model.trainable = False
        for layer in self.model.layers:
            layer.trainable = False
        output_layer = self.model.get_layer("seizure_output")
        self.embedding_model = tf.keras.Model(self.model.inputs, output_layer.input)

    @property
    def is_inference_only(self) -> bool:
        return not self.model.trainable and all(not layer.trainable for layer in self.model.layers)

    def predict(
        self,
        eeg_window: Any,
        *,
        input_channel_order: Sequence[str] | None = None,
        preprocessed: bool = False,
    ) -> dict[str, Any]:
        data = np.asarray(eeg_window)
        if data.ndim == 2:
            data = data[None, ...]
        if preprocessed:
            if data.ndim != 3 or data.shape[1:] != (self.window_samples, self.channels):
                raise ValueError("Preprocessed EEG must have shape [B, 1024, 18]")
            model_input = data.astype(np.float32)
        else:
            if data.ndim != 3 or data.shape[1:] != (self.channels, self.window_samples):
                raise ValueError("Raw EEG windows must have shape [B, 18, 1024]")
            if self.channel_order:
                supplied = tuple(input_channel_order or ())
                if supplied != self.channel_order:
                    raise ValueError("EEG montage/order does not match the configured checkpoint order")
            model_input = np.transpose(self.preprocessor.transform(data), (0, 2, 1))
        probability = np.asarray(self.model(model_input, training=False)).reshape(-1)
        embedding = np.asarray(self.embedding_model(model_input, training=False))
        clipped = np.clip(probability, 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1.0 - clipped))
        return EncoderOutput(embedding=embedding, logits=logits, probability=probability).as_dict()
