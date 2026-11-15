"""
Model Evaluation, ROC Curves & Benchmarking Script
Supports Custom CNN (Model 3) and MobileNetV2 (Model 4)

Author: Samiun Mahmud | Member 2 — Baseline CNN & Edge Vision Lead
Week 3: Added ROC curve analysis and AUC metrics

Usage:
    # Evaluate CNN
    python evaluate.py --model models/cnn_animal_classifier.keras --data_dir dataset

    # Evaluate MobileNetV2
    python evaluate.py --model models/mobilenetv2_animal_classifier.keras --data_dir dataset --model_type mobilenet

    # Generate ROC curves only
    python evaluate.py --model models/cnn_animal_classifier.keras --data_dir dataset --roc
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
except ImportError:
    tf = None

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# Default class names — alphabetical order matching dataset directories
CLASS_NAMES = ['cat', 'cow', 'dog', 'lamb', 'zebra']
CLASS_COLORS = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#9B59B6']


# ──────────────────────────── Core Evaluation ────────────────────────────────

def evaluate_model(model_path, data_dir, batch_size=32, model_type='cnn',
                   generate_roc=True, save_dir='results'):
    """
    Evaluates a saved Keras model on the test split.
    Prints classification report, confusion matrix, and optionally generates ROC curves.

    Parameters:
        model_path (str): Path to the saved .keras model file.
        data_dir (str): Path to dataset root directory.
        batch_size (int): Batch size for inference.
        model_type (str): 'cnn' or 'mobilenet' — controls data preprocessing.
        generate_roc (bool): Whether to generate and save ROC curves.
        save_dir (str): Directory to save ROC curve plots.

    Returns:
        dict: Evaluation results containing y_true, y_pred, y_proba, class_names, AUC scores.
    """
    if tf is None:
        raise ImportError("TensorFlow is required for model evaluation.")

    from data_loader import load_and_split_data

    print(f"\n[Evaluation] Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"[Evaluation] Loading test data (model_type='{model_type}')...")
    _, _, test_ds, class_names = load_and_split_data(
        data_dir=data_dir,
        batch_size=batch_size,
        model_type=model_type
    )

    y_true, y_pred, y_proba = [], [], []

    print("[Evaluation] Running predictions on test split...")
    for X_batch, y_batch in test_ds:
        proba_batch = model.predict(X_batch, verbose=0)
        pred_batch = np.argmax(proba_batch, axis=1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(pred_batch)
        y_proba.extend(proba_batch)

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_proba = np.array(y_proba)

    # ── Classification Report ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("               CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ── Confusion Matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    print("=" * 60)
    print("                  CONFUSION MATRIX")
    print("=" * 60)
    print(cm)
    print("=" * 60)

    # ── Macro AUC ──────────────────────────────────────────────────────────
    macro_auc = roc_auc_score(
        label_binarize(y_true, classes=list(range(len(class_names)))),
        y_proba,
        average='macro',
        multi_class='ovr'
    )
    print(f"\n[Evaluation] Macro-average AUC (OvR): {macro_auc:.4f}")

    results = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'class_names': class_names,
        'confusion_matrix': cm,
        'macro_auc': macro_auc,
    }

    # ── ROC Curves ────────────────────────────────────────────────────────
    if generate_roc:
        model_label = 'MobileNetV2' if model_type == 'mobilenet' else 'Custom CNN'
        suffix = 'mobilenet' if model_type == 'mobilenet' else 'cnn'
        roc_path = os.path.join(save_dir, f'roc_curves_{suffix}.png')
        plot_roc_curves(y_true, y_proba, class_names,
                        model_label=model_label, save_path=roc_path)
        results['roc_plot_path'] = roc_path

    return results


# ─────────────────────────── ROC Curve Plotting ──────────────────────────────

def plot_roc_curves(y_true, y_proba, class_names=None,
                    model_label='Model', save_path=None, figsize=(10, 8)):
    """
    Plots One-vs-Rest (OvR) ROC curves for each class plus the macro-average curve.

    For multiclass classification, we use One-vs-Rest (OvR) strategy:
    each class is treated as "positive" and all others as "negative".
    This produces one ROC curve per class, showing the trade-off between
    true positive rate (sensitivity) and false positive rate (1-specificity).

    The Area Under the Curve (AUC) for each class measures discrimination ability:
    - AUC = 1.00 → perfect separation
    - AUC = 0.50 → random classifier (diagonal line)

    Parameters:
        y_true (np.ndarray): True integer class labels, shape (N,).
        y_proba (np.ndarray): Softmax probability outputs, shape (N, num_classes).
        class_names (list): Animal class names (default: 5 animal classes).
        model_label (str): Model name shown in the plot title.
        save_path (str, optional): File path to save the PNG figure.
        figsize (tuple): Figure size in inches.

    Returns:
        dict: Per-class AUC scores + macro-average AUC.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    n_classes = len(class_names)

    # Binarize labels for OvR comparison (shape: N x n_classes)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr_dict, tpr_dict, auc_dict = {}, {}, {}

    # Per-class ROC curves
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])

    # Macro-average ROC curve (average over all classes)
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    # Per-class curves
    for i, cls_name in enumerate(class_names):
        ax.plot(
            fpr_dict[i], tpr_dict[i],
            color=CLASS_COLORS[i % len(CLASS_COLORS)],
            linewidth=2,
            label=f'{cls_name.capitalize()} (AUC = {auc_dict[i]:.3f})'
        )

    # Macro-average curve
    ax.plot(
        all_fpr, mean_tpr,
        color='navy', linestyle='--', linewidth=2.5,
        label=f'Macro-average (AUC = {macro_auc:.3f})'
    )

    # Diagonal — random classifier baseline
    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', linewidth=1.5,
            label='Random classifier (AUC = 0.500)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    ax.set_title(
        f'{model_label} — ROC Curves (One-vs-Rest, 5-Class)\n'
        f'Macro-average AUC = {macro_auc:.4f}',
        fontsize=13, fontweight='bold'
    )
    ax.legend(loc='lower right', fontsize=10, framealpha=0.90)
    ax.grid(alpha=0.25)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Evaluation] ROC curves saved -> {save_path}")

    plt.close()

    auc_dict['macro'] = macro_auc
    return auc_dict


def plot_roc_comparison(cnn_proba, mobi_proba, y_true,
                        class_names=None, save_path=None, figsize=(14, 6)):
    """
    Side-by-side ROC curve comparison: Custom CNN (left) vs MobileNetV2 (right).
    Uses macro-average curves for each model on the same axes to compare
    overall discrimination ability.

    Parameters:
        cnn_proba (np.ndarray): CNN softmax probabilities, shape (N, 5).
        mobi_proba (np.ndarray): MobileNetV2 softmax probabilities, shape (N, 5).
        y_true (np.ndarray): True integer class labels, shape (N,).
        class_names (list): Animal class names.
        save_path (str, optional): File path to save the combined figure.
        figsize (tuple): Figure size.

    Returns:
        tuple: (cnn_macro_auc, mobilenet_macro_auc)
    """
    if class_names is None:
        class_names = CLASS_NAMES

    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    def _macro_roc(y_proba):
        fpr_per, tpr_per = {}, {}
        for i in range(n_classes):
            fpr_per[i], tpr_per[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        all_fpr = np.unique(np.concatenate([fpr_per[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_per[i], tpr_per[i])
        mean_tpr /= n_classes
        return all_fpr, mean_tpr, auc(all_fpr, mean_tpr), fpr_per, tpr_per

    cnn_fpr,  cnn_tpr,  cnn_auc,  cnn_fpr_cls,  cnn_tpr_cls  = _macro_roc(cnn_proba)
    mobi_fpr, mobi_tpr, mobi_auc, mobi_fpr_cls, mobi_tpr_cls = _macro_roc(mobi_proba)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(
        'ROC Curve Comparison — Model 3 (Custom CNN) vs Model 4 (MobileNetV2)',
        fontsize=13, fontweight='bold'
    )

    for ax, fpr_cls, tpr_cls, macro_fpr, macro_tpr, macro_auc_val, label, color in [
        (axes[0], cnn_fpr_cls,  cnn_tpr_cls,  cnn_fpr,  cnn_tpr,  cnn_auc,  'Custom CNN (Model 3)',    'steelblue'),
        (axes[1], mobi_fpr_cls, mobi_tpr_cls, mobi_fpr, mobi_tpr, mobi_auc, 'MobileNetV2 (Model 4)', 'darkorange'),
    ]:
        for i, cn in enumerate(class_names):
            ax.plot(fpr_cls[i], tpr_cls[i],
                    color=CLASS_COLORS[i], linewidth=1.5, alpha=0.75,
                    label=f'{cn.capitalize()} (AUC={auc(fpr_cls[i], tpr_cls[i]):.3f})')
        ax.plot(macro_fpr, macro_tpr, color=color, linestyle='--', linewidth=2.5,
                label=f'Macro-avg (AUC={macro_auc_val:.3f})')
        ax.plot([0, 1], [0, 1], 'gray', linestyle=':', linewidth=1.2)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(f'{label}\nMacro AUC = {macro_auc_val:.4f}', fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8, framealpha=0.85)
        ax.grid(alpha=0.25)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Evaluation] ROC comparison saved -> {save_path}")

    plt.close()
    return cnn_auc, mobi_auc


# ──────────────────────────── CLI Entry Point ─────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CNN/MobileNetV2 — Classification Report, Confusion Matrix, ROC Curves"
    )
    parser.add_argument("--model", type=str,
        default="models/cnn_animal_classifier.keras",
        help="Path to trained .keras model file")
    parser.add_argument("--data_dir", type=str, default="dataset",
        help="Path to dataset root folder")
    parser.add_argument("--model_type", type=str, default="cnn",
        choices=["cnn", "mobilenet"],
        help="'cnn' or 'mobilenet' — controls data preprocessing pipeline")
    parser.add_argument("--roc", action="store_true", default=True,
        help="Generate and save ROC curve plot (default: True)")
    parser.add_argument("--save_dir", type=str, default="results",
        help="Directory to save ROC curve plots")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_dir=args.data_dir,
        model_type=args.model_type,
        generate_roc=args.roc,
        save_dir=args.save_dir
    )
