"""
CNN Training Script for Animal Image Classification
Author: Samiun Mahmud
"""

import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from model import build_cnn_model
from data_loader import load_and_split_data


def train_pipeline(data_dir, epochs=20, batch_size=32, output_model_path="models/cnn_animal_classifier.keras", log_dir="logs"):
    """
    Executes the complete CNN model training pipeline:
    - Loads & preprocesses dataset
    - Builds Sequential model architecture
    - Fits model with TensorBoard callbacks
    - Saves model and plots loss/accuracy history curves
    """
    os.makedirs(os.path.dirname(output_model_path) or ".", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = load_and_split_data(
        data_dir=data_dir,
        batch_size=batch_size
    )

    model = build_cnn_model(num_classes=len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_model_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        )
    ]

    print(f"\n[Training] Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # Save final model
    model.save(output_model_path)
    print(f"[Training] Model successfully saved to {output_model_path}")

    # Plot and save learning curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', color='teal')
    plt.plot(history.history['val_loss'], label='Val Loss', color='orange')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='green')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plot_path = "learning_curves.png"
    plt.savefig(plot_path)
    print(f"[Training] Learning curves saved to {plot_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Animal Image Classifier")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root folder")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output", type=str, default="models/cnn_animal_classifier.keras", help="Path to save trained model")
    args = parser.parse_args()

    train_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_model_path=args.output
    )
