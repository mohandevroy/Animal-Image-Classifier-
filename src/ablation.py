"""
Regularization Ablation Study — Custom CNN (Model 3)
Compares training behaviour of two CNN variants:
  - Baseline (No Regularization): No Dropout, no L2 weight decay
  - Regularized (Full):  SpatialDropout2D(0.3) + L2(1e-4) on all kernels

Author: Samiun Mahmud | Member 2 — Baseline CNN & Edge Vision Lead
Week 3: Regularization Analysis

Usage (Google Colab):
    from ablation import run_regularization_ablation, plot_ablation_curves

    history_no_reg, history_reg = run_regularization_ablation(
        train_ds, val_ds, epochs=25
    )
    plot_ablation_curves(history_no_reg, history_reg,
                         save_path='results/ablation_regularization.png')
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_regularization_ablation(train_ds, val_ds, epochs=25, num_classes=5):
    """
    Trains two CNN variants sequentially to compare regularization effect.

    Variant A — No Regularization:
        build_cnn_model(use_regularization=False)
        Dropout = 0.0, L2 = None
        Expected behaviour: fast training convergence, potential overfitting
        (val_loss diverges while train_loss keeps dropping)

    Variant B — Full Regularization (production model):
        build_cnn_model(use_regularization=True)
        SpatialDropout2D(0.3) + L2(1e-4) on Conv and Dense kernels
        Expected behaviour: slightly slower convergence but tighter train/val gap

    Parameters:
        train_ds (tf.data.Dataset): Training split yielding (images, labels).
        val_ds (tf.data.Dataset): Validation split.
        epochs (int): Number of training epochs for each variant.
        num_classes (int): Number of output classes.

    Returns:
        tuple: (history_no_reg, history_reg) — Keras History objects.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow required. Install: pip install tensorflow")

    from model import build_cnn_model

    print("\n" + "=" * 60)
    print("  ABLATION A: CNN without Regularization (Baseline)")
    print("=" * 60)
    model_no_reg = build_cnn_model(
        num_classes=num_classes,
        use_regularization=False
    )
    model_no_reg.summary()
    history_no_reg = model_no_reg.fit(
        train_ds, epochs=epochs, validation_data=val_ds, verbose=1
    )

    print("\n" + "=" * 60)
    print("  ABLATION B: CNN with Full Regularization (SpatialDropout + L2)")
    print("=" * 60)
    model_reg = build_cnn_model(
        num_classes=num_classes,
        use_regularization=True,
        spatial_dropout_rate=0.3,
        l2_reg=1e-4
    )
    model_reg.summary()
    history_reg = model_reg.fit(
        train_ds, epochs=epochs, validation_data=val_ds, verbose=1
    )

    return history_no_reg, history_reg


def plot_ablation_curves(history_no_reg, history_reg,
                         save_path=None, figsize=(16, 12)):
    """
    Plots a 2x2 comparison grid of training and validation metrics for
    the two CNN ablation variants:

    Layout:
      [Top-left]     Loss — No Regularization
      [Top-right]    Loss — With Regularization
      [Bottom-left]  Accuracy — No Regularization
      [Bottom-right] Accuracy — With Regularization

    The train/val gap visually shows how regularization reduces overfitting.

    Parameters:
        history_no_reg: Keras History object from model without regularization.
        history_reg: Keras History object from model with regularization.
        save_path (str, optional): Path to save the PNG figure.
        figsize (tuple): Figure size in inches.

    Returns:
        matplotlib.figure.Figure: The ablation comparison figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        'Regularization Ablation Study — Custom CNN (Model 3)\n'
        'Left: No Regularization   |   Right: SpatialDropout(0.3) + L2(1e-4)',
        fontsize=13, fontweight='bold', y=1.01
    )

    epochs_no = range(1, len(history_no_reg['loss']) + 1)
    epochs_re = range(1, len(history_reg['loss']) + 1)

    # ── Top row: Loss ───────────────────────────────────────────────────────
    axes[0, 0].plot(epochs_no, history_no_reg['loss'],
                    label='Train Loss', color='steelblue', linewidth=2)
    axes[0, 0].plot(epochs_no, history_no_reg['val_loss'],
                    label='Val Loss',   color='tomato',    linewidth=2, linestyle='--')
    gap0 = max(history_no_reg['val_loss']) - min(history_no_reg['train_loss']
            if 'train_loss' in history_no_reg else history_no_reg['loss'])
    axes[0, 0].fill_between(epochs_no,
                             history_no_reg['loss'],
                             history_no_reg['val_loss'],
                             alpha=0.12, color='tomato', label='Overfit gap')
    axes[0, 0].set_title('Loss — No Regularization', fontsize=11)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].plot(epochs_re, history_reg['loss'],
                    label='Train Loss', color='steelblue', linewidth=2)
    axes[0, 1].plot(epochs_re, history_reg['val_loss'],
                    label='Val Loss',   color='seagreen',  linewidth=2, linestyle='--')
    axes[0, 1].fill_between(epochs_re,
                             history_reg['loss'],
                             history_reg['val_loss'],
                             alpha=0.10, color='seagreen', label='Tight gap')
    axes[0, 1].set_title('Loss — With Regularization', fontsize=11)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.25)

    # ── Bottom row: Accuracy ─────────────────────────────────────────────────
    axes[1, 0].plot(epochs_no, history_no_reg['accuracy'],
                    label='Train Accuracy', color='steelblue', linewidth=2)
    axes[1, 0].plot(epochs_no, history_no_reg['val_accuracy'],
                    label='Val Accuracy',   color='tomato',    linewidth=2, linestyle='--')
    axes[1, 0].set_title('Accuracy — No Regularization', fontsize=11)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(epochs_re, history_reg['accuracy'],
                    label='Train Accuracy', color='steelblue', linewidth=2)
    axes[1, 1].plot(epochs_re, history_reg['val_accuracy'],
                    label='Val Accuracy',   color='seagreen',  linewidth=2, linestyle='--')
    axes[1, 1].set_title('Accuracy — With Regularization', fontsize=11)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.25)

    # ── Summary annotation ───────────────────────────────────────────────────
    final_no_val = history_no_reg['val_accuracy'][-1]
    final_re_val = history_reg['val_accuracy'][-1]
    max_no_val = max(history_no_reg['val_accuracy'])
    max_re_val = max(history_reg['val_accuracy'])

    summary = (
        f"No Regularization:   peak val_acc = {max_no_val:.3f} | final = {final_no_val:.3f}\n"
        f"With Regularization: peak val_acc = {max_re_val:.3f} | final = {final_re_val:.3f}"
    )
    fig.text(0.5, -0.01, summary, ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Ablation] Plot saved -> {save_path}")
    plt.close()
    return fig


def summarise_ablation(history_no_reg, history_reg):
    """
    Prints a compact text summary table comparing the two ablation variants.

    Parameters:
        history_no_reg (dict): History dict (Keras history.history) without regularization.
        history_reg (dict): History dict with regularization.
    """
    def gap(h):
        """Mean train-val accuracy gap across last 5 epochs (overfitting indicator)."""
        last = min(5, len(h['accuracy']))
        return np.mean(np.array(h['accuracy'][-last:]) - np.array(h['val_accuracy'][-last:]))

    print("\n" + "=" * 65)
    print("         REGULARIZATION ABLATION — SUMMARY TABLE")
    print("=" * 65)
    print(f"{'Metric':<35} {'No Reg':>10} {'With Reg':>10}")
    print("-" * 65)
    print(f"{'Peak Train Accuracy':<35} {max(history_no_reg['accuracy']):>10.4f} {max(history_reg['accuracy']):>10.4f}")
    print(f"{'Peak Val Accuracy':<35} {max(history_no_reg['val_accuracy']):>10.4f} {max(history_reg['val_accuracy']):>10.4f}")
    print(f"{'Final Val Accuracy':<35} {history_no_reg['val_accuracy'][-1]:>10.4f} {history_reg['val_accuracy'][-1]:>10.4f}")
    print(f"{'Min Val Loss':<35} {min(history_no_reg['val_loss']):>10.4f} {min(history_reg['val_loss']):>10.4f}")
    print(f"{'Train-Val Accuracy Gap (last 5 ep)':<35} {gap(history_no_reg):>10.4f} {gap(history_reg):>10.4f}")
    print("=" * 65)
    print("Note: Lower train-val gap = less overfitting = better generalization")
    print("=" * 65)
