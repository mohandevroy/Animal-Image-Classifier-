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

### Milestone 7 (October 20): Pull Request Merge, Automated Unit Testing & Visual Analytics
- **PR #1 Merged into `main`**: Submitted formal Pull Request `#1` from `Samiun-Mahmud` to `main`. Successfully resolved merge conflicts on `README.md` and merged all core CNN code into the project's primary branch (`17fe466`).
- **Automated Unit Testing (`tests/test_model.py`)**:
  - Implemented unit test suite using `unittest` to guarantee architecture validity and numerical stability.
  - Verified input tensor constraints `(None, 256, 256, 3)` and output tensor constraints `(None, 5)`.
  - Added assertion testing to ensure Softmax activation generates valid probability distributions strictly summing to 1.0.
  - Confirmed 3-block convolutional layering and dense classification head without dimension mismatches.
- **Visual Analytics Generation (`results/`)**:
  - Generated high-resolution confusion matrix heatmap (`results/confusion_matrix.png`) illustrating 100% test classification accuracy across Cat, Cow, Dog, Lamb, and Zebra.
  - Generated 20-epoch training and validation loss and accuracy progression curves (`results/training_curves.png`).
  - Embedded visual analytics into both `README.md` and `CONTRIBUTIONS_SAMIUN.md`.

### Milestone 8 (October 21): Web UI Deployment, Data Augmentation & Demo Runner
- **Interactive Streamlit Web Application (`app.py`)**:
  - Engineered an interactive web application allowing users to drag-and-drop animal images and view instant classifications.
  - Formulated real-time probability visualization bars with emoji badges for all 5 animal classes (Cat, Cow, Dog, Lamb, Zebra).
  - Integrated graceful fallback handling for quick local testing and evaluation presentations.
- **Data Augmentation Pipeline Formulation (`src/augment.py`)**:
  - Implemented `build_augmentation_pipeline()` applying `RandomFlip("horizontal")`, `RandomRotation(0.1)`, `RandomZoom(0.1)`, and `RandomContrast(0.1)` to combat potential overfitting and increase model robustness against noisy, unstandardized user photos.
  - Linked augmentation layer to the `src` package API.
- **One-Click End-to-End Pipeline Demo (`scripts/demo.py`)**:
  - Developed standalone demo script verifying preprocessing, inference formatting, and producing visual prediction reports saved directly to `results/demo_prediction.png`.

### Milestone 9 (October 22): Pull Request Merge, CI/CD Pipeline & Faculty Defense Dossier
- **PR #2 Merged into `main` (`637e6e1`)**: Successfully merged all October 20 & 21 deliverables into the project's primary branch.
- **GitHub Actions CI/CD Pipeline (`.github/workflows/ci.yml`)**:
  - Configured automated test runner executing on Ubuntu with Python 3.11.
  - Automatically installs dependencies, executes unit tests, and verifies the end-to-end pipeline on every push and pull request.
- **Packaged Test Examples (`examples/`)**:
  - Added ready-to-test sample images (`cat_test.jpg`, `cow_test.jpg`, `dog_test.jpg`, `lamb_test.jpg`, `zebra_test.jpg`) enabling instant verification without external dataset downloads.
- **Faculty Defense & Viva Dossier (`FACULTY_DEFENSE_GUIDE.md`)**:
  - Authored a comprehensive faculty viva guide detailing 2-minute elevator pitch, mathematical parameter calculations, live CLI demo instructions, and top 10 faculty viva questions with complete answers.

### Milestone 10 (October 23): 8-Model Distribution Alignment & Model 4 (MobileNetV2 Transfer Learning)
- **Role Alignment**: Formally aligned Samiun Mahmud as **Member 2: Baseline CNN & Edge Vision Lead** responsible for **Model 3 (Custom CNN)** and **Model 4 (MobileNetV2)**.
- **Model 3 Regularization (`src/model.py`)**:
  - Integrated `SpatialDropout2D(0.3)` on intermediate feature maps and L2 weight decay ($1 \times 10^{-4}$) to prevent overfitting on 100 images/class (Week 3 milestone).
  - Maintained runnable training notebook at `notebooks/02_custom_cnn_train.ipynb`.
- **Model 4: MobileNetV2 Architecture & Two-Stage Training (`src/mobilenet.py`, `notebooks/04_mobilenetv2.ipynb`)**:
  - Developed ImageNet-pretrained MobileNetV2 model wrapper with custom 5-class classification head (`GlobalAveragePooling2D` + `Dropout(0.3)` + `Dense(5)`).
  - Formulated two-stage training schedule: Phase 1 (frozen backbone, Adam lr=1e-3) and Phase 2 (unfreezing top inverted residual blocks, Adam lr=1e-5).
- **Member 2 Comparative Benchmark (`results/member2_cnn_vs_mobilenet.png`, `results/member2_model_comparison.csv`)**:
  - Benchmarked Model 3 vs Model 4 across parameters, memory footprint, latency, and accuracy:
    - Model 3 (Custom CNN): 4.21M parameters, 16.0 MB, ~12.0 ms latency, 100% test accuracy.
    - Model 4 (MobileNetV2): 2.26M parameters, 9.1 MB, ~7.5 ms latency, 100% test accuracy.
- **Streamlit App Dual-Model Selector (`app.py`)**:
  - Added sidebar architecture selector allowing live evaluation between Model 3 and Model 4 with dynamic input resolution resizing (256×256 vs 224×224).

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

### Performance Visualizations
<p align="center">
  <img src="results/training_curves.png" alt="Training & Validation Curves" width="70%" />
</p>
<p align="center">
  <img src="results/confusion_matrix.png" alt="Confusion Matrix Heatmap" width="50%" />
</p>

---

## 📂 Files Authored by Samiun Mahmud (Member 2)

```
Animal-Image-Classifier-/
├── .github/
│   └── workflows/
│       └── ci.yml                # Automated GitHub Actions CI test runner
├── .gitignore                    # Project ignore rules
├── app.py                        # Streamlit app supporting Model 3 & Model 4
├── CONTRIBUTIONS_SAMIUN.md       # Individual daily work & decision log (Milestones 1–10)
├── Dataset Drive Link            # Cloud dataset reference link
├── FACULTY_DEFENSE_GUIDE.md      # Viva presentation guide & parameter math
├── README.md                     # Project documentation & 8-model roadmap
├── requirements.txt              # Dependency specifications
├── examples/                     # Standalone sample test images for all 5 classes
│   ├── cat_test.jpg
│   ├── cow_test.jpg
│   ├── dog_test.jpg
│   ├── lamb_test.jpg
│   └── zebra_test.jpg
├── main/
│   └── cnncl.ipynb               # Baseline CNN exploration notebook
├── notebooks/                    # Standardized course notebooks
│   ├── 02_custom_cnn_train.ipynb # Model 3: Custom CNN training & regularization
│   └── 04_mobilenetv2.ipynb      # Model 4: MobileNetV2 transfer learning
├── results/
│   ├── confusion_matrix.png      # 100% test accuracy heatmap
│   ├── demo_prediction.png       # Inference pipeline verification report
│   ├── member2_cnn_vs_mobilenet.png # Model 3 vs Model 4 benchmark chart
│   ├── member2_model_comparison.csv # Consolidated Member 2 metrics table
│   └── training_curves.png       # 20-epoch loss and accuracy plots
├── scripts/
│   └── demo.py                   # One-click standalone demo runner
├── src/
│   ├── __init__.py               # Package initializer
│   ├── augment.py                # Data augmentation pipeline (flips, rotations, zooms)
│   ├── model.py                  # Regularized CNN Sequential model builder
│   ├── mobilenet.py              # MobileNetV2 transfer learning model & fine-tuning
│   ├── data_loader.py            # Dataset loading, scaling, and splitting
│   ├── train.py                  # CLI training script with callbacks
│   ├── evaluate.py               # Test evaluation and confusion matrix
│   └── predict.py                # Single image inference CLI
└── tests/
    ├── __init__.py               # Test package initializer
    └── test_model.py             # Automated unit tests for CNN architecture
```
