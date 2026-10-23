# 👨‍💻 Individual Contribution Log — Samiun Mahmud

- **Student Name**: Samiun Mahmud
- **Course**: CSE445 — Machine Learning
- **Project**: Multiclass Animal Image Classification using CNN
- **Assigned Role / Module**: **Member 2 — Custom CNN Sequential Architecture, Training Pipeline & Code Modularization**
- **Branch**: `Samiun-Mahmud`
- **Repository**: [mohandevroy/Animal-Image-Classifier-](https://github.com/mohandevroy/Animal-Image-Classifier-)

---

## 📌 Executive Summary

In this project, I am responsible for designing, training, evaluating, and modularizing the **Convolutional Neural Network (CNN) Sequential Classifier** — designated **Model 3** in the team's 8-model distribution.

The model classifies animal images into 5 categories (**Cat, Cow, Dog, Lamb, Zebra**) using a custom 3-block CNN trained from scratch on the project dataset.

Beyond exploratory experimentation in the baseline Jupyter Notebook (`main/cnncl.ipynb`), I structured the codebase into an industry-standard modular Python package (`src/`) featuring automated dataset loading, checkpointing, evaluation metrics reporting, and a command-line interface for inference.

---

## 🛠️ Key Technical & Architectural Decisions

### 1. Model Architecture (Model 3 — Custom CNN)
- **Layer Stacking**: Designed a 3-block convolutional feature extractor:
  - `Conv2D(16, (3, 3), activation='relu')` + `MaxPooling2D(2, 2)`: Captures edges, gradients, and low-level textures.
  - `Conv2D(32, (3, 3), activation='relu')` + `MaxPooling2D(2, 2)`: Captures intermediate motifs (animal ears, eyes, fur patterns).
  - `Conv2D(16, (3, 3), activation='relu')` + `MaxPooling2D(2, 2)`: Encodes high-level structural representations while constraining parameter count to prevent overfitting on ~500 images.
- **Regularization**:
  - `SpatialDropout2D(0.3)` applied after intermediate feature maps to suppress co-adaptation.
  - L2 weight decay (`1e-4`) on convolutional and dense kernels to penalize large weights.
- **Dense Classification Head**:
  - `Flatten()` followed by `Dense(256, activation='relu')` for non-linear feature combination.
  - `Dense(5, activation='softmax')` generating valid probability distributions over all 5 animal classes.

### 2. Preprocessing & Data Pipeline
- Resized input dimensions to **256 × 256 × 3**.
- Pixel intensity normalization: Scaled [0, 255] → [0.0, 1.0] using `map(lambda x, y: (x/255.0, y))`.
- Split strategy: **70% Training**, **20% Validation**, **10% Testing**.
- Performance optimization: Applied `cache()` and `prefetch(buffer_size=AUTOTUNE)` to prevent I/O bottlenecks during GPU/CPU training.

### 3. Optimization & Loss Selection
- **Optimizer**: `Adam(learning_rate=0.001)` for adaptive moment estimation and fast convergence.
- **Loss Function**: `sparse_categorical_crossentropy` to compute cross-entropy directly from integer-encoded labels without requiring one-hot expansion.

---

## 📅 Week 1 — Daily Development Log

### Day 1 (October 18): Repository Setup & Project Scaffolding
- Initialized local workspace and configured remote tracking for branch `Samiun-Mahmud`.
- Formulated `.gitignore` covering Python cache (`__pycache__`), Jupyter checkpoints, virtual environments, large datasets, and model weight binaries.
- Linked Google Drive dataset repository for team accessibility (`Dataset Drive Link` file).
- Wrote initial baseline `README.md` documenting project overview and CNN scope.

### Day 2 (October 18–19): Baseline Notebook & `src/` Package Foundation
- Built and executed the end-to-end baseline CNN pipeline in `main/cnncl.ipynb` on Google Colab.
- Verified directory-based dataset loading across all 5 animal classes; confirmed label mappings:
  - `0` → Cat, `1` → Cow, `2` → Dog, `3` → Lamb, `4` → Zebra
- Refactored monolithic notebook cells into clean, testable Python modules under `src/`:
  - `src/model.py` — `build_cnn_model()` with parameterized input shapes, class counts, and regularization (SpatialDropout + L2).
  - `src/data_loader.py` — `load_and_split_data()` with reproducible shuffling (seed 42) and 70/20/10 partitioning.
  - `src/augment.py` — `build_augmentation_pipeline()` applying RandomFlip, RandomRotation, RandomZoom, and RandomContrast to improve robustness.
  - `src/train.py` — CLI training pipeline with configurable `--epochs`, `--batch_size`, `--data_dir`, `--output` and integrated `ModelCheckpoint` & `TensorBoard` callbacks.
  - `src/evaluate.py` — Evaluation routine computing Precision, Recall, F1-score, and Confusion Matrix using `sklearn`.
  - `src/predict.py` — Single-image inference CLI accepting `--image <path>` and printing ranked confidence scores across all 5 classes.
  - `src/__init__.py` — Package initializer exporting all public functions.

### Day 3 (October 19): Dependency Management & PR #1 to `main`
- Wrote `requirements.txt` pinning all project dependencies (TensorFlow, NumPy, Matplotlib, scikit-learn, Pillow, OpenCV).
- Updated `Dataset Drive Link` with the finalized shared Google Drive folder.
- Opened and submitted **Pull Request #1** from `Samiun-Mahmud` into `main` for peer review and integration.
- PR #1 merged successfully (`17fe466`) after resolving README merge conflicts with teammates.

### Day 4 (October 20): Automated Unit Tests & Visual Performance Graphs
- **Automated Unit Testing (`tests/test_model.py`)**:
  - Implemented test suite using `unittest` to guarantee CNN architecture validity and numerical stability.
  - Tests verify: input tensor shape `(None, 256, 256, 3)`, output tensor shape `(None, 5)`, Softmax probabilities summing to 1.0, and correct 3-block convolutional layer structure.
  - CI-safe: Tests gracefully skip when TensorFlow is not installed in the environment.
- **Visual Performance Analytics (`results/`)**:
  - Generated high-resolution **confusion matrix heatmap** (`results/confusion_matrix.png`) illustrating 100% test classification accuracy across Cat, Cow, Dog, Lamb, and Zebra.
  - Generated **20-epoch training and validation** loss & accuracy curves (`results/training_curves.png`).
  - Embedded both visualizations into `README.md`.

### Day 5 (October 21–22): CI/CD Pipeline, Sample Images & Structured Training Notebook
- **GitHub Actions CI/CD Pipeline (`.github/workflows/ci.yml`)**:
  - Configured automated test runner executing on Ubuntu with Python 3.11.
  - Automatically installs dependencies, runs unit tests, and verifies the pipeline on every push and pull request to `main`.
- **Packaged Sample Test Images (`examples/`)**:
  - Added ready-to-test example images (`cat_test.jpg`, `cow_test.jpg`, `dog_test.jpg`, `lamb_test.jpg`, `zebra_test.jpg`) enabling instant offline verification without external dataset downloads.
- **Structured Training Notebook (`notebooks/02_custom_cnn_train.ipynb`)**:
  - Reorganized the exploratory notebook into a clean, section-by-section training walkthrough covering data loading, model building, training loop, regularization analysis, and evaluation.

---

## 📊 Experimental Results (Custom CNN — Model 3)

| Metric | Score | Note |
| :--- | :---: | :--- |
| **Training Accuracy** | **100%** | Reached 1.00 by epoch 18 |
| **Validation Accuracy** | **100%** | Stable convergence without divergence |
| **Test Accuracy** | **100%** | Evaluated on hold-out test set |
| **Precision (All 5 classes)** | **1.00** | Cat, Cow, Dog, Lamb, Zebra |
| **Recall (All 5 classes)** | **1.00** | Zero misclassifications |
| **F1-Score (All 5 classes)** | **1.00** | Harmonic mean of precision & recall |

### Performance Visualizations
<p align="center">
  <img src="results/training_curves.png" alt="Training & Validation Curves" width="70%" />
</p>
<p align="center">
  <img src="results/confusion_matrix.png" alt="Confusion Matrix Heatmap" width="50%" />
</p>

---

## 📂 Files Authored by Samiun Mahmud (Week 1 Scope)

```
Animal-Image-Classifier-/
├── .github/
│   └── workflows/
│       └── ci.yml                # Automated GitHub Actions CI test runner
├── .gitignore                    # Project ignore rules
├── CONTRIBUTIONS_SAMIUN.md       # Individual daily work & decision log
├── Dataset Drive Link            # Cloud dataset reference link
├── README.md                     # Project documentation & CNN architecture overview
├── requirements.txt              # Dependency specifications
├── examples/                     # Standalone sample test images for all 5 classes
│   ├── cat_test.jpg
│   ├── cow_test.jpg
│   ├── dog_test.jpg
│   ├── lamb_test.jpg
│   └── zebra_test.jpg
├── main/
│   └── cnncl.ipynb               # Baseline CNN exploration notebook (Google Colab)
├── notebooks/
│   └── 02_custom_cnn_train.ipynb # Model 3: Custom CNN training & regularization walkthrough
├── results/
│   ├── confusion_matrix.png      # 100% test accuracy heatmap (5-class)
│   └── training_curves.png       # 20-epoch loss and accuracy progression plots
├── src/
│   ├── __init__.py               # Package initializer exporting CNN functions
│   ├── augment.py                # Data augmentation pipeline (flips, rotations, zooms)
│   ├── model.py                  # Regularized CNN Sequential model builder
│   ├── data_loader.py            # Dataset loading, scaling, and train/val/test splitting
│   ├── train.py                  # CLI training script with ModelCheckpoint & TensorBoard
│   ├── evaluate.py               # Test evaluation: accuracy, F1, confusion matrix
│   └── predict.py                # Single image inference CLI
└── tests/
    ├── __init__.py               # Test package initializer
    └── test_model.py             # Automated unit tests for CNN architecture & output shapes
```
