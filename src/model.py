"""
CNN Model Architecture for Animal Image Classification
Author: Samiun Mahmud
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, SpatialDropout2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def build_cnn_model(
    input_shape=(256, 256, 3),
    num_classes=5,
    learning_rate=0.001,
    use_regularization=True,
    spatial_dropout_rate=0.3,
    l2_reg=1e-4
):
    """
    Builds and compiles a Convolutional Neural Network (CNN)
    using Keras Sequential API for multiclass animal classification.
    Supports Spatial Dropout (0.3) and L2 weight decay for overfitting control.

    Parameters:
        input_shape (tuple): Dimensions of input images (height, width, channels).
        num_classes (int): Number of target classes (default: 5 - cat, cow, dog, lamb, zebra).
        learning_rate (float): Learning rate for the Adam optimizer.
        use_regularization (bool): Whether to apply Spatial Dropout and L2 regularization (Week 3).
        spatial_dropout_rate (float): Dropout probability for 2D feature maps.
        l2_reg (float): L2 weight decay penalty factor.

    Returns:
        tf.keras.Model: Compiled Sequential CNN model.
    """
    reg = l2(l2_reg) if use_regularization else None

    layers_list = [
        # Block 1: Low-level edges & gradients
        Conv2D(16, (3, 3), strides=1, activation='relu', kernel_regularizer=reg, input_shape=input_shape, name="conv_1"),
        MaxPooling2D(pool_size=(2, 2), name="pool_1"),

        # Block 2: Intermediate motifs & textures
        Conv2D(32, (3, 3), strides=1, activation='relu', kernel_regularizer=reg, name="conv_2"),
        MaxPooling2D(pool_size=(2, 2), name="pool_2"),
    ]

    # Spatial Dropout applied to intermediate feature maps (Week 3 Regularization)
    if use_regularization and spatial_dropout_rate > 0:
        layers_list.append(SpatialDropout2D(spatial_dropout_rate, name="spatial_dropout"))

    layers_list.extend([
        # Block 3: Higher-level structural representations
        Conv2D(16, (3, 3), strides=1, activation='relu', kernel_regularizer=reg, name="conv_3"),
        MaxPooling2D(pool_size=(2, 2), name="pool_3"),

        # Classification Head
        Flatten(name="flatten"),
        Dense(256, activation='relu', kernel_regularizer=reg, name="dense_features"),
        Dropout(0.3 if use_regularization else 0.0, name="dense_dropout"),
        Dense(num_classes, activation='softmax', name="output_probabilities")
    ])

    model = Sequential(layers_list, name="Animal_Classifier_CNN")

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
