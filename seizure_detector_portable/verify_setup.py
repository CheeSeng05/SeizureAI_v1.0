from pathlib import Path
import sys

print("=" * 60)
print("EEG PROJECT ENVIRONMENT CHECK")
print("=" * 60)
print("Python:", sys.executable)

try:
    import tensorflow as tf
    import keras
    print("TensorFlow:", tf.__version__)
    print("Keras:", keras.__version__)
    print("CUDA build:", tf.test.is_built_with_cuda())
    print("GPUs:", tf.config.list_physical_devices("GPU"))
except Exception as exc:
    print("TensorFlow/Keras check FAILED:", repr(exc))
    raise

model_path = Path("seizure_detector_model.keras")

if not model_path.exists():
    raise FileNotFoundError(
        "seizure_detector_model.keras is not in the project directory."
    )

from tensorflow.keras.models import load_model

model = load_model(model_path, compile=False)
print("Model loaded: YES")
print("Model input:", model.input_shape)
print("Model output:", model.output_shape)

if tuple(model.input_shape[1:]) != (1024, 18):
    raise RuntimeError(
        f"Unexpected model input shape: {model.input_shape}"
    )

if tuple(model.output_shape[1:]) != (1,):
    raise RuntimeError(
        f"Unexpected model output shape: {model.output_shape}"
    )

print("MODEL CHECK: PASS")
