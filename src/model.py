"""
CNN Model Architecture for Animal Image Classification
Author: Samiun Mahmud
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam


def build_cnn_model(input_shape=(256, 256, 3), num_classes=5, learning_rate=0.001):
    """
    Builds and compiles a 3-layer Convolutional Neural Network (CNN)
    using Keras Sequential API for multiclass image classification.

    Parameters:
        input_shape (tuple): Dimensions of input images (height, width, channels).
        num_classes (int): Number of target classes (default: 5 - cat, cow, dog, lamb, zebra).
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        tf.keras.Model: Compiled Sequential CNN model.
    """
    model = Sequential([
        # Block 1: Feature Extraction
        Conv2D(16, (3, 3), strides=1, activation='relu', input_shape=input_shape, name="conv_1"),
        MaxPooling2D(pool_size=(2, 2), name="pool_1"),

        # Block 2: Intermediate Representation
        Conv2D(32, (3, 3), strides=1, activation='relu', name="conv_2"),
        MaxPooling2D(pool_size=(2, 2), name="pool_2"),

        # Block 3: Higher-Level Features
        Conv2D(16, (3, 3), strides=1, activation='relu', name="conv_3"),
        MaxPooling2D(pool_size=(2, 2), name="pool_3"),

        # Classification Head
        Flatten(name="flatten"),
        Dense(256, activation='relu', name="dense_features"),
        Dense(num_classes, activation='softmax', name="output_probabilities")
    ], name="Animal_Classifier_CNN")

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
