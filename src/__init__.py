"""
Multiclass Animal Image Classification Package
Week 1: Custom CNN Baseline (Model 3)
Author: Samiun Mahmud | Member 2 — Baseline CNN & Edge Vision Lead
"""

from .model import build_cnn_model
from .data_loader import load_and_split_data
from .augment import build_augmentation_pipeline
from .predict import predict_image

__all__ = [
    "build_cnn_model",
    "load_and_split_data",
    "build_augmentation_pipeline",
    "predict_image"
]
