"""
Dataset Loader & Preprocessor for Animal Image Classification
Author: Samiun Mahmud
"""

import os
import tensorflow as tf
from pathlib import Path

# Alphabetical classes mapping
CLASS_NAMES = ['cat', 'cow', 'dog', 'lamb', 'zebra']


def load_and_split_data(data_dir, image_size=(256, 256), batch_size=32, train_ratio=0.7, val_ratio=0.2):
    """
    Loads images from class directories, normalizes pixels to [0, 1],
    and splits into train, validation, and test datasets.

    Parameters:
        data_dir (str or Path): Path to dataset folder containing class subdirectories.
        image_size (tuple): Target image resolution (height, width).
        batch_size (int): Batch size for training and evaluation.
        train_ratio (float): Fraction of data for training (default: 0.70).
        val_ratio (float): Fraction of data for validation (default: 0.20).

    Returns:
        tuple: (train_ds, val_ds, test_ds, class_names)
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found at: {data_dir}")

    # Load dataset from directory
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    class_names = dataset.class_names
    print(f"[DataLoader] Detected classes: {class_names}")

    # Normalize pixel values to [0, 1]
    normalized_ds = dataset.map(lambda x, y: (x / 255.0, y))

    total_batches = len(normalized_ds)
    train_size = int(total_batches * train_ratio)
    val_size = int(total_batches * val_ratio)
    test_size = total_batches - train_size - val_size

    train_ds = normalized_ds.take(train_size)
    val_ds = normalized_ds.skip(train_size).take(val_size)
    test_ds = normalized_ds.skip(train_size + val_size).take(test_size)

    # Prefetch for performance
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"[DataLoader] Batches - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    return train_ds, val_ds, test_ds, class_names
