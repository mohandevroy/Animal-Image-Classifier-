"""
Advanced Data Augmentation Pipeline for Animal Image Classification
Implements two augmentation backends:
  - Keras built-in layers (Week 1 baseline — always available)
  - Albumentations (Week 3 advanced — richer transforms, requires pip install)

Author: Samiun Mahmud | Member 2 — Baseline CNN & Edge Vision Lead
Week 3: Advanced Augmentation Study

Albumentations Install (Google Colab):
    !pip install albumentations -q

Usage:
    # Keras baseline pipeline (always works)
    aug = build_augmentation_pipeline()

    # Albumentations advanced pipeline
    aug_fn = build_albumentations_pipeline(image_size=(256, 256), mode='medium')

    # Wrap into tf.data compatible map function
    tf_aug = make_tf_augmentation_fn(aug_fn, image_size=(256, 256))
    train_ds = train_ds.map(tf_aug, num_parallel_calls=tf.data.AUTOTUNE)
"""

# ─────────────────────────── Keras Augmentation Pipeline (Week 1) ────────────────────────────

def build_augmentation_pipeline(image_size=(256, 256)):
    """
    Constructs a Keras Sequential data augmentation pipeline.
    Uses only built-in Keras layers — no external dependencies.

    Augmentations applied:
      - RandomFlip (horizontal): mirrors left-right to increase viewpoint variety
      - RandomRotation (±10%): simulates slight camera tilt or animal head turn
      - RandomZoom (±10%): simulates varying distance from the camera
      - RandomContrast (±10%): simulates lighting and exposure changes

    Parameters:
        image_size (tuple): Target image resolution (height, width). Not used by Keras
            random layers directly but kept for API consistency with albumentations pipeline.

    Returns:
        tf.keras.Sequential: Augmentation layer pipeline callable on batches.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Sequential
    except ImportError:
        raise ImportError(
            "TensorFlow is required for the Keras augmentation pipeline. "
            "Install it with: pip install tensorflow"
        )

    augmentation_pipeline = Sequential([
        layers.RandomFlip("horizontal", name="aug_horizontal_flip"),
        layers.RandomRotation(0.10, fill_mode="nearest", name="aug_rotation"),
        layers.RandomZoom(0.10, fill_mode="nearest", name="aug_zoom"),
        layers.RandomContrast(0.10, name="aug_contrast"),
    ], name="Animal_Data_Augmenter_Keras")

    return augmentation_pipeline


def apply_augmentation_to_dataset(dataset):
    """
    Applies the Keras augmentation pipeline to a tf.data.Dataset during training.

    Parameters:
        dataset (tf.data.Dataset): Dataset yielding (images, labels) batches.

    Returns:
        tf.data.Dataset: Augmented dataset with AUTOTUNE prefetching.
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    augmenter = build_augmentation_pipeline()
    augmented_ds = dataset.map(
        lambda x, y: (augmenter(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return augmented_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# ─────────────────────── Albumentations Advanced Pipeline (Week 3) ───────────────────────────

# Augmentation preset configurations — ordered by intensity
ALBUMENTATIONS_PRESETS = {
    'light': {
        'description': 'Conservative augmentations — minimal distortion, good for high-accuracy baselines',
        'HorizontalFlip_p':     0.5,
        'Rotate_limit':         10,
        'Rotate_p':             0.4,
        'ColorJitter_brightness': 0.1,
        'ColorJitter_contrast':   0.1,
        'ColorJitter_saturation': 0.1,
        'ColorJitter_hue':        0.0,
        'ColorJitter_p':          0.3,
        'GaussNoise_p':           0.0,
        'HueSaturation_p':        0.0,
        'ElasticTransform_p':     0.0,
        'Cutout_p':               0.0,
    },
    'medium': {
        'description': 'Balanced augmentations — good for 500–2000 image datasets like ours',
        'HorizontalFlip_p':     0.5,
        'Rotate_limit':         20,
        'Rotate_p':             0.5,
        'ColorJitter_brightness': 0.2,
        'ColorJitter_contrast':   0.2,
        'ColorJitter_saturation': 0.2,
        'ColorJitter_hue':        0.05,
        'ColorJitter_p':          0.5,
        'GaussNoise_p':           0.3,
        'HueSaturation_p':        0.3,
        'ElasticTransform_p':     0.0,
        'Cutout_p':               0.2,
    },
    'strong': {
        'description': 'Aggressive augmentations — maximises data variety for small datasets',
        'HorizontalFlip_p':     0.5,
        'Rotate_limit':         35,
        'Rotate_p':             0.6,
        'ColorJitter_brightness': 0.3,
        'ColorJitter_contrast':   0.3,
        'ColorJitter_saturation': 0.3,
        'ColorJitter_hue':        0.1,
        'ColorJitter_p':          0.6,
        'GaussNoise_p':           0.4,
        'HueSaturation_p':        0.4,
        'ElasticTransform_p':     0.2,
        'Cutout_p':               0.3,
    },
}


def build_albumentations_pipeline(image_size=(256, 256), mode='medium'):
    """
    Constructs an Albumentations augmentation pipeline with configurable intensity.

    Why Albumentations over Keras built-ins?
    - Richer transforms: HueSaturation, ElasticTransform, GaussianNoise, CoarseDropout
    - Faster on CPU (native NumPy, no TF graph overhead)
    - Pixel-level + spatial transforms in a single composable API
    - Standard in research and Kaggle competitions for small-dataset augmentation

    Transforms applied (mode='medium'):
      - HorizontalFlip:       Mirror left-right (p=0.5)
      - Rotate:               Random rotation up to ±20 deg (p=0.5)
      - ColorJitter:          Brightness, contrast, saturation, hue variation (p=0.5)
      - GaussNoise:           Additive Gaussian noise for sensor noise simulation (p=0.3)
      - HueSaturationValue:   Hue/saturation/brightness shift in HSV space (p=0.3)
      - CoarseDropout:        Randomly mask rectangular patches (regularization) (p=0.2)

    Parameters:
        image_size (tuple): Image resolution (height, width) — used for CoarseDropout sizing.
        mode (str): Augmentation intensity. One of: 'light', 'medium', 'strong'.
            See ALBUMENTATIONS_PRESETS for per-transform probabilities.

    Returns:
        albumentations.Compose: Composed transform callable on {'image': np.ndarray} dicts.

    Raises:
        ImportError: If albumentations is not installed.
        ValueError: If an unknown mode is passed.
    """
    try:
        import albumentations as A
    except ImportError:
        raise ImportError(
            "Albumentations is not installed.\n"
            "Install it in Colab with: !pip install albumentations -q\n"
            "Or locally with: pip install albumentations"
        )

    if mode not in ALBUMENTATIONS_PRESETS:
        raise ValueError(
            f"Unknown augmentation mode: '{mode}'. "
            f"Choose from: {list(ALBUMENTATIONS_PRESETS.keys())}"
        )

    cfg = ALBUMENTATIONS_PRESETS[mode]
    h, w = image_size

    # CoarseDropout (Cutout) patch size: 10% of image dimensions
    patch_h = max(1, int(h * 0.10))
    patch_w = max(1, int(w * 0.10))

    transforms = [
        A.HorizontalFlip(p=cfg['HorizontalFlip_p']),

        A.Rotate(
            limit=cfg['Rotate_limit'],
            border_mode=0,     # 0 = cv2.BORDER_CONSTANT (fill with black)
            p=cfg['Rotate_p']
        ),

        A.ColorJitter(
            brightness=cfg['ColorJitter_brightness'],
            contrast=cfg['ColorJitter_contrast'],
            saturation=cfg['ColorJitter_saturation'],
            hue=cfg['ColorJitter_hue'],
            p=cfg['ColorJitter_p']
        ),
    ]

    if cfg['GaussNoise_p'] > 0:
        transforms.append(
            A.GaussNoise(p=cfg['GaussNoise_p'])
        )

    if cfg['HueSaturation_p'] > 0:
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=15,
                p=cfg['HueSaturation_p']
            )
        )

    if cfg['ElasticTransform_p'] > 0:
        transforms.append(
            A.ElasticTransform(
                alpha=1.0,
                sigma=50,
                p=cfg['ElasticTransform_p']
            )
        )

    if cfg['Cutout_p'] > 0:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(patch_h // 2, patch_h),
                hole_width_range=(patch_w // 2, patch_w),
                fill=0,
                p=cfg['Cutout_p']
            )
        )

    pipeline = A.Compose(transforms)
    print(f"[Augment] Albumentations pipeline built — mode='{mode}' | {len(transforms)} transforms")
    return pipeline


def make_tf_augmentation_fn(albumentations_pipeline, image_size=(256, 256)):
    """
    Wraps an Albumentations pipeline into a tf.data.Dataset-compatible map function.

    Albumentations operates on NumPy arrays (uint8 HxWxC), while tf.data pipelines
    use float32 tensors. This wrapper handles the conversion safely using tf.numpy_function.

    Parameters:
        albumentations_pipeline: Albumentations Compose object from build_albumentations_pipeline().
        image_size (tuple): Target image resolution (height, width).

    Returns:
        callable: A function (image_tensor, label) -> (augmented_tensor, label)
            compatible with tf.data.Dataset.map().

    Example:
        aug_pipeline = build_albumentations_pipeline(image_size=(224, 224), mode='medium')
        aug_fn = make_tf_augmentation_fn(aug_pipeline, image_size=(224, 224))
        train_ds = train_ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    """
    try:
        import tensorflow as tf
        import numpy as np
    except ImportError:
        raise ImportError("TensorFlow and NumPy are required.")

    h, w = image_size

    def _augment_numpy(image_np):
        """Apply Albumentations to a single HxWxC uint8 numpy image."""
        image_uint8 = (image_np * 255).astype('uint8')
        augmented = albumentations_pipeline(image=image_uint8)
        result = augmented['image'].astype('float32') / 255.0
        return result

    def augment_batch(images, labels):
        """Map over a batch applying _augment_numpy to each image."""
        augmented_images = tf.map_fn(
            lambda img: tf.numpy_function(
                _augment_numpy, [img], tf.float32
            ),
            images,
            fn_output_signature=tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32)
        )
        augmented_images.set_shape([None, h, w, 3])
        return augmented_images, labels

    return augment_batch


def compare_augmentation_backends(image_path, image_size=(256, 256)):
    """
    Visual comparison utility: applies both Keras and Albumentations augmentation
    to a single test image and displays side-by-side sample outputs (2x4 grid).

    Useful for verifying augmentation effects before training.

    Parameters:
        image_path (str): Path to a sample input image (JPG/PNG).
        image_size (tuple): Target image resolution (height, width).

    Returns:
        matplotlib.figure.Figure: Figure object showing augmented samples.
    """
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    # Load and resize original image
    img = Image.open(image_path).convert('RGB').resize(image_size[::-1])  # PIL is (W,H)
    img_np = np.array(img).astype('float32') / 255.0
    img_uint8 = np.array(img)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle(
        'Augmentation Comparison: Keras (top row) vs Albumentations/medium (bottom row)',
        fontsize=12, fontweight='bold'
    )

    # Top row: original + Keras augmented samples
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original', fontsize=10)
    axes[0, 0].axis('off')

    try:
        import tensorflow as tf
        keras_aug = build_augmentation_pipeline(image_size)
        for i in range(1, 5):
            batch = tf.expand_dims(tf.constant(img_np), 0)
            aug_batch = keras_aug(batch, training=True)
            axes[0, i].imshow(aug_batch[0].numpy())
            axes[0, i].set_title(f'Keras #{i}', fontsize=10)
            axes[0, i].axis('off')
    except ImportError:
        for i in range(1, 5):
            axes[0, i].text(0.5, 0.5, 'TF not available', ha='center', va='center')
            axes[0, i].axis('off')

    # Bottom row: original + Albumentations augmented samples
    axes[1, 0].imshow(img_np)
    axes[1, 0].set_title('Original', fontsize=10)
    axes[1, 0].axis('off')

    try:
        albu_pipeline = build_albumentations_pipeline(image_size=image_size, mode='medium')
        for i in range(1, 5):
            aug = albu_pipeline(image=img_uint8)
            aug_img = aug['image'].astype('float32') / 255.0
            axes[1, i].imshow(aug_img)
            axes[1, i].set_title(f'Albumentations #{i}', fontsize=10)
            axes[1, i].axis('off')
    except ImportError:
        for i in range(1, 5):
            axes[1, i].text(0.5, 0.5, 'albumentations\nnot installed', ha='center', va='center')
            axes[1, i].axis('off')

    plt.tight_layout()
    return fig
