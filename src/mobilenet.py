"""
MobileNetV2 Transfer Learning Architecture for Animal Image Classification
Model 4: MobileNetV2 (Pretrained on ImageNet with Frozen Feature Extractor)
Author: Samiun Mahmud | Member 2 — Baseline CNN & Edge Vision Lead
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_mobilenet_model(
    input_shape=(224, 224, 3),
    num_classes=5,
    learning_rate=0.001,
    freeze_backbone=True,
    dropout_rate=0.3
):
    """
    Builds and compiles a transfer learning classifier using MobileNetV2.

    Parameters:
        input_shape (tuple): Expected input dimensions (default: 224x224x3).
        num_classes (int): Number of target classes (default: 5 - Cat, Cow, Dog, Lamb, Zebra).
        learning_rate (float): Learning rate for Adam optimizer (Phase 1 default: 1e-3).
        freeze_backbone (bool): Whether to freeze pretrained ImageNet weights (Phase 1).
        dropout_rate (float): Dropout probability before final classification head.

    Returns:
        tf.keras.Model: Compiled Keras functional Model.
    """
    inputs = Input(shape=input_shape, name="input_image")

    # Load MobileNetV2 backbone pretrained on ImageNet, excluding original 1000-class head
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    # Phase 1: Freeze feature extractor weights to preserve pretrained representations
    base_model.trainable = not freeze_backbone

    # Forward pass through backbone
    x = base_model(inputs, training=False)

    # Custom Classification Head for 5-Class Animal Classification
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name="head_dropout")(x)

    outputs = Dense(num_classes, activation="softmax", name="output_probabilities")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Animal_Classifier_MobileNetV2")

    # Compile with adaptive optimizer and sparse cross-entropy loss
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def unfreeze_mobilenet_top_layers(model, num_layers=30, learning_rate=1e-5):
    """
    Unfreezes the top N layers of the MobileNetV2 backbone for fine-tuning (Phase 2).

    Parameters:
        model (tf.keras.Model): The MobileNetV2 model instance.
        num_layers (int): Number of top layers to unfreeze from the end of the backbone.
        learning_rate (float): Lower learning rate for fine-tuning (default: 1e-5).

    Returns:
        tf.keras.Model: Recompiled model with partially trainable backbone.
    """
    # Locate base MobileNetV2 layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("MobileNetV2 base model layer not found in the given model.")

    base_model.trainable = True

    # Freeze all layers except the last `num_layers`
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
    for layer in base_model.layers[-num_layers:]:
        layer.trainable = True

    # Recompile with smaller learning rate to prevent catastrophic forgetting
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_mobilenet_model()
    model.summary()
