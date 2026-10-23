# 🎓 Faculty Defense & Viva Presentation Guide

- **Student Name**: Samiun Mahmud
- **Course**: CSE445 — Machine Learning
- **Project**: Multiclass Animal Image Classification (8-Model Benchmark)
- **Assigned Role**: **Member 2 — Baseline CNN & Edge Vision Lead**
- **Assigned Models**:
  - **Model 3**: Custom CNN (Trained from scratch baseline with Spatial Dropout & L2 decay)
  - **Model 4**: MobileNetV2 (Pretrained ImageNet transfer learning with fine-tuning)
- **Branch**: `Samiun-Mahmud` (or `feat/cnn-and-mobilenet`)
- **Repository**: [mohandevroy/Animal-Image-Classifier-](https://github.com/mohandevroy/Animal-Image-Classifier-)

---

## 🎯 1. Two-Minute Elevator Pitch (Say This to Your Faculty)
> *"Good morning/afternoon, Sir. In our group project's 8-model architecture, I served as **Member 2: Baseline CNN & Edge Vision Lead** on branch **`Samiun-Mahmud`**.*  
> *I was responsible for two complementary models across the machine learning spectrum:*  
> *1. **Model 3 (Custom CNN)**: Engineered from scratch with a 3-block convolutional hierarchy, Spatial Dropout (0.3), and L2 weight decay ($1\times 10^{-4}$) to prevent overfitting on 100 images per class, reaching 100% test accuracy.*  
> *2. **Model 4 (MobileNetV2)**: Implemented an edge-optimized transfer learning pipeline utilizing ImageNet pretrained weights. I executed a two-stage training schedule: freezing the 154-layer backbone for feature extraction, followed by unfreezing the top inverted residual blocks with a $10\times$ lower learning rate ($1\times 10^{-5}$) for fine-tuning.*  
> *In our benchmark comparison, MobileNetV2 reduced parameter count by **46%** (2.26M vs 4.21M) and inference latency to just **7.5 ms** while maintaining 100% accuracy. Both models are fully integrated into our interactive Streamlit web app."*

---

## 🧠 2. Mathematical Architecture & Parameter Breakdown

If your faculty asks about the model layer-by-layer:

| Layer | Type | Output Shape | Parameters Formula | Trainable Params | Purpose |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Input** | `InputLayer` | `(256, 256, 3)` | — | `0` | Normalized RGB animal image |
| **Block 1** | `Conv2D(16, 3×3)` | `(256, 256, 16)` | $(3 \times 3 \times 3 + 1) \times 16$ | `448` | Extracts low-level edges & gradients |
| | `MaxPooling2D(2×2)` | `(128, 128, 16)` | — | `0` | Spatial downsampling (reduces compute) |
| **Block 2** | `Conv2D(32, 3×3)` | `(128, 128, 32)` | $(3 \times 3 \times 16 + 1) \times 32$ | `4,640` | Captures animal textures & motifs |
| | `MaxPooling2D(2×2)` | `(64, 64, 32)` | — | `0` | Spatial downsampling |
| **Block 3** | `Conv2D(16, 3×3)` | `(64, 64, 16)` | $(3 \times 3 \times 32 + 1) \times 16$ | `4,624` | Encodes high-level shapes (ears, snouts) |
| | `MaxPooling2D(2×2)` | `(32, 32, 16)` | — | `0` | Downsampling to $32 \times 32$ feature map |
| **Head** | `Flatten` | `(16384)` | $32 \times 32 \times 16$ | `0` | Vectorizes feature maps |
| | `Dense(256, ReLU)` | `(256)` | $16,384 \times 256 + 256$ | `4,194,560` | Non-linear dense feature combination |
| | `Dense(5, Softmax)`| `(5)` | $256 \times 5 + 5$ | `1,285` | Outputs probability distribution over classes |
| **Total** | — | — | — | **~4,205,557** | **~16.0 MB model size** |

---

## 💻 3. Live Demonstration Commands

Run these commands live in terminal to present your work:

### 1. Run the One-Click Demo Runner (Instant verification)
```bash
python scripts/demo.py
```
*Outputs preprocessing validation, prediction confidence, and saves visual graph to `results/demo_prediction.png`.*

### 2. Execute Automated Unit Tests (Proves Reliability)
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```
*Executes all 5 automated unit tests validating input dimensions, output dimensions, layer sequence, and Softmax probability constraints ($\sum P_i = 1.0$).*

### 3. Launch the Interactive Web Application
```bash
streamlit run app.py
```
*Opens an interactive browser UI where the professor can drag and drop any image from `examples/` and view real-time confidence scores.*

### 4. Run CLI Inference on Sample Images
```bash
python src/predict.py --image examples/cat_test.jpg
python src/predict.py --image examples/dog_test.jpg
```

---

## ❓ 4. Top 10 Faculty Viva Questions & Perfect Answers

#### Q1: Why did you choose a 3-block CNN Sequential architecture instead of a pre-trained ResNet or VGG?
> *"Pretrained models like ResNet-50 or VGG-16 contain 25 to 138 million parameters. On a 500-image dataset, heavy architectures easily memorize training samples and overfit. Our lightweight 3-block CNN contains ~4.2M parameters, trains in under 2 minutes, and generalizes with 100% test accuracy without computational bloat."*

#### Q2: What is the purpose of Data Augmentation in your pipeline?
> *"In `src/augment.py`, we implemented `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)`, and `RandomContrast(0.1)`. Because animal photos taken with phones vary in orientation, lighting, and framing, data augmentation artificially synthesizes these real-world conditions, acting as an effective regularizer."*

#### Q3: Why `sparse_categorical_crossentropy` instead of `categorical_crossentropy`?
> *"TensorFlow's `image_dataset_from_directory` generates integer labels (`0, 1, 2, 3, 4`). `sparse_categorical_crossentropy` computes the loss directly on integer indices, avoiding the memory and computation overhead of creating one-hot vectors for every batch."*

#### Q4: Why divide pixel values by 255.0?
> *"Raw 8-bit image pixels range from $[0, 255]$. Normalizing by $255.0$ rescales them to $[0.0, 1.0]$. This prevents vanishing or exploding gradients, ensures numerical stability, and accelerates Adam optimizer convergence."*

#### Q5: How did you ensure your 100% accuracy wasn't due to data leakage or overfitting?
> *"We applied strict split isolation: 70% Train, 20% Validation, and 10% Test. The test split of 50 images was strictly held out and never seen during training or callback monitoring. In addition, our training vs. validation curves strictly tracked each other without diverging."*

#### Q6: How does your CNN model compare with the group's YOLOv11 model?
> *"YOLOv11 Nano is an object detector adapted for classification and achieved 94% accuracy. Our custom CNN achieved 100% accuracy because it is specifically tailored to full-frame centered animal classification with $256 \times 256$ inputs, avoiding the multi-scale detection anchor overhead of YOLO."*

#### Q7: What software engineering best practices did you implement?
> *"I refactored the project from a raw Jupyter notebook into a modular package (`src/`), added automated unit tests (`tests/`), implemented CI/CD via GitHub Actions (`.github/workflows/ci.yml`), established clean `.gitignore` rules, and built an interactive Streamlit web application (`app.py`)."*

#### Q8: What does the Softmax activation function guarantee mathematically?
> *"Softmax exponentiates logits $z_i$ and normalizes them: $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$. This guarantees two crucial mathematical properties: every output is strictly positive ($P_i > 0$), and all probabilities sum exactly to $1.0$ ($\sum P_i = 1.0$). We explicitly tested this guarantee in `tests/test_model.py`."*

#### Q9: What is the role of MaxPooling layers?
> *"MaxPooling with a $2 \times 2$ window extracts the dominant feature from each region while halving the spatial dimensions ($256 \rightarrow 128 \rightarrow 64 \rightarrow 32$). This provides translation invariance and exponentially reduces the number of operations in subsequent layers."*

#### Q10: How do you prove your individual contribution on GitHub?
> *"All my commits are tracked on branch `Samiun-Mahmud` with atomic commits spanning from October 18 to October 22. Pull Request #1 and Pull Request #2 were reviewed and merged into `main`. Every technical milestone and rationale is documented in `CONTRIBUTIONS_SAMIUN.md`."*
