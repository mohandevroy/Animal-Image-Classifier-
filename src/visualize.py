"""
Feature Space Visualization — PCA & t-SNE 2D Projections
Extracts deep feature embeddings from trained CNN or MobileNetV2 models
and plots 2D projections to visualize inter-class separability.

Author: Samiun Mahmud | Member 2 — Baseline CNN & Edge Vision Lead
Week 3: Feature Analysis & Regularization Study

Usage (Google Colab):
    from visualize import extract_features, plot_pca, plot_tsne, plot_pca_and_tsne

    # Extract features from CNN model
    features, labels = extract_features(model, test_ds, layer_name='flatten')

    # Extract features from MobileNetV2 model
    features, labels = extract_features(model, test_ds, layer_name='global_avg_pool')

    # Plot PCA
    plot_pca(features, labels, class_names, save_path='results/pca_cnn.png')

    # Plot t-SNE
    plot_tsne(features, labels, class_names, save_path='results/tsne_cnn.png')

    # Side-by-side PCA + t-SNE
    plot_pca_and_tsne(features, labels, class_names, title='Custom CNN', save_path='results/feature_space_cnn.png')
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')    # Non-interactive backend — safe in Colab / server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 5-class animal colour palette — consistent across all plots
CLASS_COLORS = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12', '#9B59B6']

# Default animal class names (alphabetical, matching dataset directory order)
CLASS_NAMES = ['cat', 'cow', 'dog', 'lamb', 'zebra']


def extract_features(model, dataset, layer_name=None):
    """
    Extracts intermediate feature embeddings from a trained Keras model
    by creating a sub-model that outputs at the specified layer.

    Parameters:
        model (tf.keras.Model): Trained Keras model (CNN or MobileNetV2).
        dataset (tf.data.Dataset): Dataset yielding (images, labels) batches.
        layer_name (str, optional): Name of the layer to extract features from.
            - Custom CNN: 'flatten' or 'dense_features'
            - MobileNetV2: 'global_avg_pool'
            If None, uses the second-to-last layer (before final softmax head).

    Returns:
        tuple: (features np.ndarray of shape (N, D), labels np.ndarray of shape (N,))
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required to extract model features. "
            "Run this function in a Colab environment with TensorFlow installed."
        )

    # Build feature extractor sub-model
    if layer_name is not None:
        feature_layer = model.get_layer(layer_name)
        feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=feature_layer.output,
            name="feature_extractor"
        )
    else:
        # Default: use the penultimate layer (before softmax output)
        feature_extractor = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output,
            name="feature_extractor"
        )

    all_features = []
    all_labels = []

    for X_batch, y_batch in dataset:
        batch_features = feature_extractor.predict(X_batch, verbose=0)
        all_features.append(batch_features)
        all_labels.append(y_batch.numpy())

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    print(f"[Visualize] Extracted features: {features.shape} | Labels: {labels.shape}")
    return features, labels


def plot_pca(features, labels, class_names=None, title='PCA — Feature Space',
             save_path=None, figsize=(8, 6)):
    """
    Projects high-dimensional features to 2D using Principal Component Analysis (PCA)
    and plots a scatter plot coloured by animal class.

    PCA finds the two directions of maximum variance in the feature space,
    showing global structure and class separability.

    Parameters:
        features (np.ndarray): Feature matrix, shape (N, D).
        labels (np.ndarray): Integer class labels, shape (N,).
        class_names (list): List of class name strings (default: ['cat','cow','dog','lamb','zebra']).
        title (str): Plot title.
        save_path (str, optional): Path to save the PNG figure. If None, displays only.
        figsize (tuple): Figure size in inches.

    Returns:
        np.ndarray: 2D PCA-projected coordinates, shape (N, 2).
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Fit PCA and project to 2D
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features)

    explained_var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=figsize)

    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=CLASS_COLORS[cls_idx % len(CLASS_COLORS)],
            label=cls_name.capitalize(),
            alpha=0.80,
            s=80,
            edgecolors='white',
            linewidths=0.5
        )

    ax.set_xlabel(f'PC1 ({explained_var[0]:.1f}% variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({explained_var[1]:.1f}% variance)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.85)
    ax.grid(alpha=0.25)

    total_var = explained_var[0] + explained_var[1]
    ax.text(0.02, 0.02, f'Total variance explained: {total_var:.1f}%',
            transform=ax.transAxes, fontsize=9, color='gray', va='bottom')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualize] PCA plot saved -> {save_path}")

    plt.close()
    return features_2d


def plot_tsne(features, labels, class_names=None, title='t-SNE — Feature Space',
              save_path=None, figsize=(8, 6), perplexity=30, n_iter=1000):
    """
    Projects high-dimensional features to 2D using t-SNE (t-distributed Stochastic
    Neighbor Embedding) and plots a scatter plot coloured by animal class.

    t-SNE preserves local neighbourhood structure — tightly clustered classes
    indicate the model has learned well-separated feature representations.

    Parameters:
        features (np.ndarray): Feature matrix, shape (N, D).
        labels (np.ndarray): Integer class labels, shape (N,).
        class_names (list): List of class name strings.
        title (str): Plot title.
        save_path (str, optional): Path to save the PNG figure.
        figsize (tuple): Figure size in inches.
        perplexity (int): t-SNE perplexity (5–50; use ~N/5 for small datasets).
        n_iter (int): Number of t-SNE optimization iterations.

    Returns:
        np.ndarray: 2D t-SNE-projected coordinates, shape (N, 2).
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # For small datasets, reduce perplexity if needed (must be < N/3)
    effective_perplexity = min(perplexity, max(5, len(features) // 4))

    # First reduce dimensionality with PCA to speed up t-SNE on high-dim features
    if features.shape[1] > 50:
        # Cap components to min(50, n_samples-1) to satisfy PCA constraints
        n_pre = min(50, features.shape[0] - 1)
        pca_pre = PCA(n_components=n_pre, random_state=42)
        features_reduced = pca_pre.fit_transform(features)
        print(f"[Visualize] Pre-PCA: {features.shape[1]}D -> {n_pre}D before t-SNE")
    else:
        features_reduced = features

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        max_iter=n_iter,
        random_state=42,
        learning_rate='auto',
        init='pca'
    )
    features_2d = tsne.fit_transform(features_reduced)

    fig, ax = plt.subplots(figsize=figsize)

    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        ax.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=CLASS_COLORS[cls_idx % len(CLASS_COLORS)],
            label=cls_name.capitalize(),
            alpha=0.80,
            s=80,
            edgecolors='white',
            linewidths=0.5
        )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.85)
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.02, f'Perplexity={effective_perplexity} | Iterations={n_iter}',
            transform=ax.transAxes, fontsize=9, color='gray', va='bottom')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualize] t-SNE plot saved -> {save_path}")

    plt.close()
    return features_2d


def plot_pca_and_tsne(features, labels, class_names=None,
                      title='Feature Space Analysis',
                      save_path=None, figsize=(16, 6),
                      perplexity=30, n_iter=1000):
    """
    Generates a side-by-side PCA (left) and t-SNE (right) feature space plot
    in a single figure — useful for comparing global vs local structure.

    Parameters:
        features (np.ndarray): Feature matrix, shape (N, D).
        labels (np.ndarray): Integer class labels, shape (N,).
        class_names (list): List of class name strings.
        title (str): Overall figure title (displayed above both subplots).
        save_path (str, optional): Path to save the PNG figure.
        figsize (tuple): Figure size in inches (default: wide format for two panels).
        perplexity (int): t-SNE perplexity parameter.
        n_iter (int): t-SNE optimization iterations.

    Returns:
        tuple: (pca_coords, tsne_coords) — both np.ndarray of shape (N, 2).
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # ── PCA projection ──────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(features)
    explained_var = pca.explained_variance_ratio_ * 100

    # ── t-SNE projection ─────────────────────────────────────────────────────
    effective_perplexity = min(perplexity, max(5, len(features) // 4))
    if features.shape[1] > 50:
        # Cap components to min(50, n_samples-1) to satisfy PCA constraints
        n_pre = min(50, features.shape[0] - 1)
        pca_pre = PCA(n_components=n_pre, random_state=42)
        features_reduced = pca_pre.fit_transform(features)
    else:
        features_reduced = features

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        max_iter=n_iter,
        random_state=42,
        learning_rate='auto',
        init='pca'
    )
    tsne_2d = tsne.fit_transform(features_reduced)

    # ── Side-by-side plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    legend_handles = [
        mpatches.Patch(color=CLASS_COLORS[i], label=cn.capitalize())
        for i, cn in enumerate(class_names)
    ]

    # PCA panel
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        axes[0].scatter(
            pca_2d[mask, 0], pca_2d[mask, 1],
            c=CLASS_COLORS[cls_idx % len(CLASS_COLORS)],
            alpha=0.80, s=80, edgecolors='white', linewidths=0.5
        )
    axes[0].set_title('PCA — 2D Feature Projection\n(Global Structure)', fontsize=12)
    axes[0].set_xlabel(f'PC1 ({explained_var[0]:.1f}% var)', fontsize=10)
    axes[0].set_ylabel(f'PC2 ({explained_var[1]:.1f}% var)', fontsize=10)
    axes[0].grid(alpha=0.25)
    axes[0].text(0.02, 0.02, f'Total variance: {sum(explained_var):.1f}%',
                 transform=axes[0].transAxes, fontsize=8, color='gray', va='bottom')

    # t-SNE panel
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        if mask.sum() == 0:
            continue
        axes[1].scatter(
            tsne_2d[mask, 0], tsne_2d[mask, 1],
            c=CLASS_COLORS[cls_idx % len(CLASS_COLORS)],
            alpha=0.80, s=80, edgecolors='white', linewidths=0.5
        )
    axes[1].set_title('t-SNE — 2D Feature Projection\n(Local Cluster Structure)', fontsize=12)
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=10)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=10)
    axes[1].grid(alpha=0.25)
    axes[1].text(0.02, 0.02, f'Perplexity={effective_perplexity}',
                 transform=axes[1].transAxes, fontsize=8, color='gray', va='bottom')

    # Shared legend below the plots
    fig.legend(handles=legend_handles, loc='lower center', ncol=5,
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Visualize] PCA+t-SNE side-by-side plot saved -> {save_path}")

    plt.close()
    return pca_2d, tsne_2d
