"""
Multiclass Animal Image Classification Package
Part 2: CNN Sequential Classifier
Author: Samiun Mahmud
"""

from .model import build_cnn_model
from .data_loader import load_and_split_data
from .predict import predict_image
from .augment import build_augmentation_pipeline

__all__ = ["build_cnn_model", "load_and_split_data", "predict_image", "build_augmentation_pipeline"]
