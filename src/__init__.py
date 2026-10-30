"""
Multiclass Animal Image Classification Package
Week 1: Custom CNN Baseline (Model 3)
Week 2: MobileNetV2 Transfer Learning (Model 4)
Author: Samiun Mahmud | Member 2 — Baseline CNN & Edge Vision Lead
"""

# Lazy imports — only load when TensorFlow is available
# This prevents import errors in environments without TensorFlow (e.g., CI runners)
try:
    from .model import build_cnn_model
    from .mobilenet import build_mobilenet_model, unfreeze_mobilenet_top_layers
    from .data_loader import load_and_split_data
    from .augment import build_augmentation_pipeline
    from .predict import predict_image

    __all__ = [
        "build_cnn_model",
        "build_mobilenet_model",
        "unfreeze_mobilenet_top_layers",
        "load_and_split_data",
        "build_augmentation_pipeline",
        "predict_image"
    ]
except ImportError:
    # TensorFlow not installed — package still importable for structural checks
    __all__ = []
