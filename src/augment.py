"""
Data Augmentation Pipeline for Animal Image Classification
Author: Samiun Mahmud
Course: CSE445 - Machine Learning
"""

import tensorflow as tf
from tensorflow.keras import layers, Sequential


def build_augmentation_pipeline(image_size=(256, 256)):
    """
    Constructs a Keras Sequential data augmentation pipeline
    to improve generalization and prevent overfitting on small animal datasets.

    Augmentations applied:
    - Random horizontal flipping
    - Random rotation (up to 10%)
    - Random zoom (up to 10%)
    - Random contrast variation (up to 10%)

    Parameters:
        image_size (tuple): Target image resolution (height, width).

    Returns:
        tf.keras.Sequential: Augmentation layer pipeline.
    """
    augmentation_pipeline = Sequential([
        layers.RandomFlip("horizontal", name="aug_horizontal_flip"),
        layers.RandomRotation(0.10, fill_mode="nearest", name="aug_rotation"),
        layers.RandomZoom(0.10, fill_mode="nearest", name="aug_zoom"),
        layers.RandomContrast(0.10, name="aug_contrast")
    ], name="Animal_Data_Augmenter")

    return augmentation_pipeline


def apply_augmentation_to_dataset(dataset):
    """
    Applies the augmentation pipeline to a tf.data.Dataset during training.

    Parameters:
        dataset (tf.data.Dataset): Dataset yielding (images, labels) batches.

    Returns:
        tf.data.Dataset: Augmented dataset with AUTOTUNE prefetching.
    """
    augmenter = build_augmentation_pipeline()
    augmented_ds = dataset.map(
        lambda x, y: (augmenter(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return augmented_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
