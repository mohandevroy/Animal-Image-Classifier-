"""
MobileNetV2 Pretrained Transfer Learning Model for Animal Image Classification
Model 4 in 8-Model Distribution (Member 2: Baseline CNN & Edge Vision)
Author: Samiun Mahmud
Course: CSE445 - Machine Learning
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam


def build_mobilenetv2_model(
    input_shape=(224, 224, 3),
    num_classes=5,
    freeze_backbone=True,
    fine_tune_at=100,
    learning_rate=1e-3
):
    """
    Constructs a lightweight MobileNetV2 transfer learning model
    pre-trained on ImageNet, adapted for 5-class animal classification.

    Parameters:
        input_shape (tuple): Input image resolution (default: (224, 224, 3)).
        num_classes (int): Number of target classes (default: 5 - Cat, Cow, Dog, Lamb, Zebra).
        freeze_backbone (bool): Whether to freeze all backbone layers for initial feature extraction.
        fine_tune_at (int): Layer index from which to begin fine-tuning during Phase 2.
        learning_rate (float): Learning rate for optimization.

    Returns:
        tf.keras.Model: Compiled MobileNetV2 classification model.
    """
    # 1. Base Pretrained Backbone (ImageNet weights, exclude 1000-class head)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    if freeze_backbone:
        base_model.trainable = False
        print(f"[MobileNetV2] Feature Extraction Phase: All {len(base_model.layers)} backbone layers frozen.")
    else:
        # Fine-tuning phase: Unfreeze top inverted residual blocks
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        trainable_count = len([l for l in base_model.layers if l.trainable])
        print(f"[MobileNetV2] Fine-Tuning Phase: Top {trainable_count} layers unfrozen starting at index {fine_tune_at}.")

    # 2. Custom Classification Head
    inputs = layers.Input(shape=input_shape, name="input_tensor")

    # MobileNetV2 expects inputs in range [-1, 1]; apply preprocessing layer
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)

    # Dimensionality reduction & regularization
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="animal_predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="Animal_Classifier_MobileNetV2")

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def unfreeze_for_fine_tuning(model, base_model, fine_tune_at=100, fine_tune_lr=1e-5):
    """
    Unfreezes the top inverted residual blocks of MobileNetV2
    and re-compiles with a 10x-100x lower learning rate for stage-2 fine-tuning.
    """
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    unfrozen_layers = len([l for l in base_model.layers if l.trainable])
    print(f"[MobileNetV2] Unfroze top {unfrozen_layers} layers for Stage 2 fine-tuning.")

    model.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    model, base = build_mobilenetv2_model()
    model.summary()
