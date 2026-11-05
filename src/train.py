"""
Unified Training Script for Animal Image Classification
Supports Model 3 (Custom CNN) and Model 4 (MobileNetV2 Transfer Learning)
Author: Samiun Mahmud | Member 2 — Baseline CNN & Edge Vision Lead

Usage:
    # Train Custom CNN (Model 3):
    python train.py --model cnn --data_dir /path/to/dataset --epochs 20

    # Train MobileNetV2 (Model 4):
    python train.py --model mobilenet --data_dir /path/to/dataset --epochs 15
"""

import os
import argparse
import matplotlib
matplotlib.use('Agg')    # Non-interactive backend — safe for Colab / server environments
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path

from model import build_cnn_model
from mobilenet import build_mobilenet_model
from data_loader import load_and_split_data


# ─────────────────────────── Helper: Save Learning Curves ───────────────────────────

def save_learning_curves(history, output_path, model_label):
    """
    Plots and saves training vs validation loss & accuracy curves.

    Parameters:
        history: Keras History object returned by model.fit().
        output_path (str): File path to save the PNG figure.
        model_label (str): Display label for the chart title (e.g. 'Custom CNN' or 'MobileNetV2').
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_label} — Training History", fontsize=14, fontweight='bold')

    # Loss subplot
    axes[0].plot(history.history['loss'], label='Train Loss', color='teal', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='orange', linewidth=2)
    axes[0].set_title('Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy subplot
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', color='steelblue', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', color='seagreen', linewidth=2)
    axes[1].set_title('Accuracy Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Training] Learning curves saved → {output_path}")


# ──────────────────────────── CNN Training Pipeline ─────────────────────────────────

def train_cnn_pipeline(data_dir, epochs=20, batch_size=32,
                        output_model_path="models/cnn_animal_classifier.keras",
                        log_dir="logs/cnn"):
    """
    Executes the full Custom CNN training pipeline (Model 3).
    - Loads data at 256×256 with [0, 1] normalization
    - Builds 3-block CNN with SpatialDropout + L2 regularization
    - Trains with ModelCheckpoint & TensorBoard callbacks
    - Saves model and learning curve plot

    Parameters:
        data_dir (str): Path to the dataset root directory.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        output_model_path (str): Path to save the best model checkpoint.
        log_dir (str): TensorBoard log directory.

    Returns:
        tuple: (model, history)
    """
    os.makedirs(os.path.dirname(output_model_path) or '.', exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Model 3: Custom CNN — Training Pipeline")
    print("=" * 60)

    train_ds, val_ds, test_ds, class_names = load_and_split_data(
        data_dir=data_dir,
        batch_size=batch_size,
        model_type='cnn'          # 256x256, x/255.0 normalization
    )

    model = build_cnn_model(num_classes=len(class_names))
    model.summary()

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_model_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    print(f"\n[Training] CNN — {epochs} epochs | Batch: {batch_size} | Classes: {class_names}")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks
    )

    model.save(output_model_path)
    print(f"[Training] Model saved → {output_model_path}")

    save_learning_curves(history, "results/training_curves_cnn.png", "Custom CNN (Model 3)")

    return model, history


# ─────────────────────── MobileNetV2 Training Pipeline ──────────────────────────────

def train_mobilenet_pipeline(data_dir, epochs=15, batch_size=32,
                              output_model_path="models/mobilenetv2_animal_classifier.keras",
                              log_dir="logs/mobilenet",
                              phase2_epochs=10):
    """
    Executes the two-phase MobileNetV2 transfer learning pipeline (Model 4).

    Phase 1 — Head Training (frozen backbone):
        - Loads data at 224×224 with MobileNetV2 preprocessing ([-1, 1])
        - Builds MobileNetV2 with frozen ImageNet weights + custom classification head
        - Trains only the head for `epochs` epochs

    Phase 2 — Fine-Tuning (top 30 layers unfrozen):
        - Unfreezes the last 30 layers of the MobileNetV2 backbone
        - Recompiles with 10× lower learning rate (1e-5) to prevent catastrophic forgetting
        - Continues training for `phase2_epochs` more epochs

    Parameters:
        data_dir (str): Path to the dataset root directory.
        epochs (int): Phase 1 (head-only) training epochs.
        batch_size (int): Batch size.
        output_model_path (str): Path to save the best model checkpoint.
        log_dir (str): TensorBoard log directory.
        phase2_epochs (int): Additional fine-tuning epochs in Phase 2.

    Returns:
        tuple: (model, phase1_history, phase2_history)
    """
    os.makedirs(os.path.dirname(output_model_path) or '.', exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Model 4: MobileNetV2 Transfer Learning — Training Pipeline")
    print("=" * 60)

    train_ds, val_ds, test_ds, class_names = load_and_split_data(
        data_dir=data_dir,
        batch_size=batch_size,
        model_type='mobilenet'    # 224x224, mobilenet_v2.preprocess_input [-1, 1]
    )

    model = build_mobilenet_model(
        num_classes=len(class_names),
        freeze_backbone=True,
        learning_rate=1e-3
    )
    model.summary()

    # ── Phase 1: Train classification head only ──────────────────────────────────
    print(f"\n[Phase 1] Training classification head — {epochs} epochs (backbone frozen)")

    phase1_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, 'phase1'), histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_model_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    phase1_history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=phase1_callbacks
    )

    best_phase1_val_acc = max(phase1_history.history['val_accuracy'])
    print(f"\n[Phase 1] Best val_accuracy: {best_phase1_val_acc:.4f}")

    # ── Phase 2: Fine-tune top 30 MobileNetV2 layers ────────────────────────────
    print(f"\n[Phase 2] Fine-tuning top 30 backbone layers — {phase2_epochs} epochs (LR=1e-5)")

    from mobilenet import unfreeze_mobilenet_top_layers
    model = unfreeze_mobilenet_top_layers(model, num_layers=30, learning_rate=1e-5)

    phase2_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir, 'phase2'), histogram_freq=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_model_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    phase2_history = model.fit(
        train_ds,
        epochs=phase2_epochs,
        validation_data=val_ds,
        callbacks=phase2_callbacks
    )

    model.save(output_model_path)
    print(f"[Training] Model saved → {output_model_path}")

    # Merge Phase 1 + Phase 2 history for combined learning curve
    combined_history_dict = {
        'loss': phase1_history.history['loss'] + phase2_history.history['loss'],
        'val_loss': phase1_history.history['val_loss'] + phase2_history.history['val_loss'],
        'accuracy': phase1_history.history['accuracy'] + phase2_history.history['accuracy'],
        'val_accuracy': phase1_history.history['val_accuracy'] + phase2_history.history['val_accuracy'],
    }

    # Create a simple object to mimic history.history
    class _HistoryWrapper:
        def __init__(self, hist_dict):
            self.history = hist_dict

    combined = _HistoryWrapper(combined_history_dict)
    save_learning_curves(combined, "results/mobilenet_curves.png", "MobileNetV2 (Model 4) — Phase 1+2")

    return model, phase1_history, phase2_history


# ─────────────────────────────── CLI Entry Point ────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Animal Image Classifier — Custom CNN or MobileNetV2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model cnn --data_dir dataset --epochs 20
  python train.py --model mobilenet --data_dir dataset --epochs 15 --phase2_epochs 10
        """
    )
    parser.add_argument(
        "--model", type=str, default="cnn",
        choices=["cnn", "mobilenet"],
        help="Model architecture to train: 'cnn' (Custom CNN) or 'mobilenet' (MobileNetV2)"
    )
    parser.add_argument("--data_dir", type=str, default="dataset",
        help="Path to dataset root folder containing class subdirectories")
    parser.add_argument("--epochs", type=int, default=20,
        help="Number of training epochs (Phase 1 for MobileNetV2)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output", type=str, default=None,
        help="Path to save trained model (.keras). Auto-named if not specified.")
    parser.add_argument("--phase2_epochs", type=int, default=10,
        help="MobileNetV2 only: Number of Phase 2 fine-tuning epochs")

    args = parser.parse_args()

    if args.model == "cnn":
        output = args.output or "models/cnn_animal_classifier.keras"
        train_cnn_pipeline(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_model_path=output
        )
    elif args.model == "mobilenet":
        output = args.output or "models/mobilenetv2_animal_classifier.keras"
        train_mobilenet_pipeline(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_model_path=output,
            phase2_epochs=args.phase2_epochs
        )
