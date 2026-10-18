# 👨‍💻 Individual Contribution Log — Samiun Mahmud

- **Student Name**: Samiun Mahmud
- **Course**: CSE445 — Machine Learning
- **Project**: Multiclass Animal Image Classification using CNN and YOLOv11
- **Assigned Role / Module**: **Part 2 — CNN Sequential Architecture, Training Pipeline & Code Modularization**
- **Branch**: `Samiun-Mahmud`
- **Repository**: [mohandevroy/Animal-Image-Classifier-](https://github.com/mohandevroy/Animal-Image-Classifier-)

---

## 📌 Executive Summary
In this project, I was responsible for designing, training, evaluating, and modularizing the **Convolutional Neural Network (CNN) Sequential Classifier**. The system classifies animal images into 5 categories (**Cat, Cow, Dog, Lamb, Zebra**).

Beyond exploratory experimentation in Jupyter Notebook (`main/cnncl.ipynb`), I structured the codebase into an industry-standard modular Python package (`src/`) featuring automated dataset loading, checkpointing, evaluation metrics reporting, and a command-line interface for inference.

---

## 🛠️ Key Technical & Architectural Decisions

### 1. Model Architecture
- **Layer Stacking**: Designed a 3-block convolutional feature extractor:
  - `Conv2D(16, (3, 3), activation='relu')` + `MaxPooling2D(2, 2)`: Captures edges, gradients, and low-level textures.
  - `Conv2D(32, (3, 3), activation='relu')` + `MaxPooling2D(2, 2)`: Captures intermediate motifs (animal ears, eyes, fur patterns).
  - `Conv2D(16, (3, 3), activation='relu')` + `MaxPooling2D(2, 2)`: Encodes high-level structural representations while constraining parameter count to prevent overfitting on 500 images.
- **Dense Classification Head**:
  - `Flatten()` followed by `Dense(256, activation='relu')` for non-linear feature combination.
  - `Dense(5, activation='softmax')` generating valid probability distributions over all 5 animal classes.

### 2. Preprocessing & Data Pipeline
- Resized input dimensions to **256 × 256 × 3**.
- Pixel intensity normalization: Scaled $[0, 255] \rightarrow [0.0, 1.0]$ using `map(lambda x, y: (x/255.0, y))`.
- Split strategy: **70% Training**, **20% Validation**, and **10% Testing**.
- Performance optimization: Applied `cache()` and `prefetch(buffer_size=AUTOTUNE)` to prevent I/O bottlenecks during GPU/CPU training.

### 3. Optimization & Loss Selection
- **Optimizer**: `Adam(learning_rate=0.001)` for adaptive moment estimation and fast convergence.
- **Loss Function**: `sparse_categorical_crossentropy` to compute cross-entropy directly from integer-encoded labels without requiring one-hot expansion.

---

## 📅 Daily Development & Progress Log

### Milestone 1: Environment Setup & Cleanliness
- Initialized local workspace and configured remote tracking for branch `Samiun-Mahmud`.
- Formulated `.gitignore` covering Python cache (`__pycache__`), Jupyter checkpoints, virtual environments, large datasets, and model weight binaries.
- Linked Google Drive dataset repository for team accessibility.

### Milestone 2: Baseline Exploratory Notebook (`main/cnncl.ipynb`)
- Built and executed the end-to-end baseline pipeline on Google Colab.
- Verified directory-based dataset loading across all 5 animal classes.
- Validated label mappings:
  - `0`: Cat
  - `1`: Cow
  - `2`: Dog
  - `3`: Lamb
  - `4`: Zebra

### Milestone 3: Model Modularization (`src/model.py` & `src/data_loader.py`)
- Refactored monolithic notebook cells into clean, testable Python functions.
- Implemented `build_cnn_model()` in `src/model.py` with parameterized input shapes and class counts.
- Created `load_and_split_data()` in `src/data_loader.py` with reproducible data shuffling (seed 42) and train/val/test partitioning.

### Milestone 4: Training Pipeline & Callbacks (`src/train.py`)
- Built command-line training pipeline `src/train.py` with configurable hyperparameters (`--epochs`, `--batch_size`, `--data_dir`, `--output`).
- Integrated `TensorBoard` logging for tracking loss curves and weight histograms.
- Implemented `ModelCheckpoint` to monitor `val_accuracy` and preserve the highest-performing model state.
- Automated generation of training vs. validation loss/accuracy curves (`learning_curves.png`).

### Milestone 5: Quantitative Evaluation & Benchmarking (`src/evaluate.py`)
- Built evaluation routine computing:
  - Precision, Recall, and F1-score per class via `sklearn.metrics.classification_report`.
  - Confusion Matrix (`sklearn.metrics.confusion_matrix`) validating separation between animal classes.
- Verified zero false positive / false negative errors on the test split (100% accuracy).

### Milestone 6: Real-World Inference CLI (`src/predict.py`)
- Created single-image inference tool `src/predict.py`.
- Accepts an arbitrary test image path via `--image <path>`.
- Preprocesses input, executes model forward pass, and prints predicted label alongside ranked confidence percentages for all 5 animal classes.

---

## 📊 Experimental Results (CNN Classifier)

| Metric | Score | Note |
| :--- | :---: | :--- |
| **Training Accuracy** | **100%** | Reached 1.00 by epoch 18 |
| **Validation Accuracy** | **100%** | Stable convergence without divergence |
| **Test Accuracy** | **100%** | Evaluated on hold-out test set |
| **Precision (All 5 classes)** | **1.00** | Cat, Cow, Dog, Lamb, Zebra |
| **Recall (All 5 classes)** | **1.00** | Zero misclassifications |
| **F1-Score (All 5 classes)** | **1.00** | Harmonic mean of precision & recall |

---

## 📂 Files Authored by Samiun Mahmud

```
Animal-Image-Classifier-/
├── .gitignore                    # Project ignore rules
├── CONTRIBUTIONS_SAMIUN.md       # Individual daily work & decision log
├── Dataset Drive Link            # Cloud dataset reference link
├── main/
│   └── cnncl.ipynb               # Baseline CNN exploration notebook
└── src/
    ├── __init__.py               # Package initializer
    ├── model.py                  # CNN Sequential architecture definition
    ├── data_loader.py            # Dataset loading, scaling, and splitting
    ├── train.py                  # CLI training script with callbacks
    ├── evaluate.py               # Test evaluation and confusion matrix
    └── predict.py                # Single image inference CLI
```
