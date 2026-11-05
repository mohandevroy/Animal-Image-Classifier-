"""
Dataset Loader & Preprocessor for Animal Image Classification
Author: Samiun Mahmud
Week 1: Custom CNN Data Pipeline (256x256, [0, 1] normalization)
Week 2: MobileNetV2 Transfer Learning Pipeline (224x224, [-1, 1] preprocessing)
"""

import os
from pathlib import Path

# Safe TensorFlow import for environments without GPU / TF (e.g., CI runners)
try:
    import tensorflow as tf
except ImportError:
    tf = None

# Alphabetical classes mapping
CLASS_NAMES = ['cat', 'cow', 'dog', 'lamb', 'zebra']

# Architecture-specific standard image resolutions
MODEL_IMAGE_SIZES = {
    'cnn': (256, 256),
    'mobilenet': (224, 224),
    'mobilenetv2': (224, 224),
}


def get_default_image_size(model_type='cnn'):
    """
    Returns the standard input image dimensions (height, width) for the specified model.

    Parameters:
        model_type (str): Target model architecture ('cnn' or 'mobilenet').

    Returns:
        tuple: (height, width) e.g., (256, 256) for CNN, (224, 224) for MobileNetV2.
    """
    key = str(model_type).strip().lower()
    return MODEL_IMAGE_SIZES.get(key, (256, 256))


def get_preprocessing_fn(mode='standard'):
    """
    Returns the appropriate preprocessing transformation for the given mode.

    Modes:
    - 'standard' / 'cnn': scales [0, 255] -> [0.0, 1.0] via x / 255.0
    - 'mobilenet' / 'mobilenetv2': scales [0, 255] -> [-1.0, 1.0] via MobileNetV2 preprocess_input

    Parameters:
        mode (str): Preprocessing mode ('standard' or 'mobilenet').

    Returns:
        callable: Function (x, y) -> (preprocessed_x, y) for tf.data.Dataset.map.
    """
    if tf is None:
        raise ImportError(
            "TensorFlow is required to build preprocessing transformations. "
            "Please ensure TensorFlow is installed in your environment."
        )

    mode_key = str(mode).strip().lower()
    if mode_key in ('mobilenet', 'mobilenetv2'):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return lambda x, y: (preprocess_input(x), y)
    elif mode_key in ('standard', 'cnn'):
        return lambda x, y: (x / 255.0, y)
    else:
        raise ValueError(
            f"Unknown preprocessing mode: '{mode}'. "
            f"Supported options: 'standard' (or 'cnn'), 'mobilenet' (or 'mobilenetv2')."
        )


def load_and_split_data(
    data_dir,
    image_size=None,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.2,
    model_type='cnn',
    preprocessing=None
):
    """
    Loads images from class directories, normalizes/preprocesses pixels according
    to the target model architecture, and splits into train, validation, and test datasets.

    Supported Model Workflows:
    - Custom CNN (Model 3):
        - Default resolution: (256, 256)
        - Preprocessing: 'standard' ([0, 1] normalization via x / 255.0)
    - MobileNetV2 Transfer Learning (Model 4):
        - Default resolution: (224, 224)
        - Preprocessing: 'mobilenet' ([-1, 1] scaling via mobilenet_v2.preprocess_input)

    Parameters:
        data_dir (str or Path): Path to dataset folder containing class subdirectories.
        image_size (tuple, optional): Target image resolution (height, width).
            Defaults to (224, 224) for MobileNetV2, (256, 256) for Custom CNN.
        batch_size (int): Batch size for training and evaluation.
        train_ratio (float): Fraction of data for training (default: 0.70).
        val_ratio (float): Fraction of data for validation (default: 0.20).
        model_type (str): Target model architecture ('cnn' or 'mobilenet'). Default: 'cnn'.
        preprocessing (str, optional): Preprocessing scheme ('standard', 'mobilenet').
            If None, automatically inferred from model_type.

    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names)
    """
    if tf is None:
        raise ImportError(
            "TensorFlow is required to load and split datasets. "
            "Please ensure TensorFlow is installed in your environment."
        )

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found at: {data_dir}")

    # Resolve target image size based on model type if not explicitly passed
    if image_size is None:
        image_size = get_default_image_size(model_type)

    # Invert/infer preprocessing mode if not explicitly passed
    if preprocessing is None:
        preprocessing = 'mobilenet' if str(model_type).strip().lower() in ('mobilenet', 'mobilenetv2') else 'standard'

    # Load raw dataset from directory
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    class_names = dataset.class_names
    print(f"[DataLoader] Detected classes: {class_names}")
    print(f"[DataLoader] Model target: '{model_type}' | Resolution: {image_size} | Preprocessing: '{preprocessing}'")

    # Apply architecture-specific preprocessing
    preprocess_fn = get_preprocessing_fn(preprocessing)
    processed_ds = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Partition into Train / Validation / Test sets
    total_batches = len(processed_ds)
    train_size = int(total_batches * train_ratio)
    val_size = int(total_batches * val_ratio)
    test_size = total_batches - train_size - val_size

    train_ds = processed_ds.take(train_size)
    val_ds = processed_ds.skip(train_size).take(val_size)
    test_ds = processed_ds.skip(train_size + val_size).take(test_size)

    # Prefetch and cache for optimal GPU / CPU I/O throughput
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"[DataLoader] Batches - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    return train_ds, val_ds, test_ds, class_names
