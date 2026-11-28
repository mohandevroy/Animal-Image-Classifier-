# Five-Class Image Classification Using ResNet50 and DenseNet121
WEEK 2 & 3
## Project Overview

This project implements and compares two pretrained convolutional neural network (CNN) architectures, **ResNet50** and **DenseNet121**, for five-class image classification.

The models were fine-tuned using transfer learning with ImageNet-pretrained weights. Both models were trained and evaluated using the same dataset split and evaluation procedure.

**Best-performing model in this run: ResNet50**, based on weighted F1-score.

## Dataset

The dataset contains five image classes:

- Cat
- Cow
- Dog
- Lamb
- Zebra

### Dataset structure

```text
dataset/
├── cat/
├── cow/
├── dog/
├── lamb/
└── zebra/
```

### Class distribution

| Class | Number of images |
|---|---:|
| Cat | 100 |
| Cow | 100 |
| Dog | 100 |
| Lamb | 213 |
| Zebra | 100 |
| **Total** | **613** |

The dataset is imbalanced because the Lamb class contains more images than the other classes.

## Dataset Split

A stratified split was used to preserve the class proportions across the subsets.

| Subset | Images | Percentage |
|---|---:|---:|
| Training | 429 | 70% |
| Validation | 92 | 15% |
| Testing | 92 | 15% |
| **Total** | **613** | **100%** |

The split used `random_state=42`.

## Data Preprocessing

Images were resized to **224 × 224** pixels and normalized using ImageNet mean and standard deviation.

Training images used the following augmentation:

- Random horizontal flip
- Random rotation
- Color jitter
- Resize
- ImageNet normalization

Validation and test images used resizing and ImageNet normalization without random augmentation.

## Models

### 1. ResNet50

ResNet50 is a 50-layer residual CNN. Its pretrained ImageNet weights were loaded, and the final fully connected layer was replaced with a classifier containing five output classes.

### 2. DenseNet121

DenseNet121 is a densely connected CNN architecture. Its pretrained ImageNet weights were loaded, and its final classifier was replaced with a five-class output layer.

Both models were trained separately using the same training and validation subsets.

## Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Environment | Google Colab |
| GPU | NVIDIA Tesla T4 |
| Input image size | 224 × 224 |
| Batch size | 16 |
| Epochs | 15 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Loss function | CrossEntropyLoss |
| Learning-rate scheduler | ReduceLROnPlateau |
| Pretrained weights | ImageNet |

The best checkpoint for each model was selected based on validation loss.

## Test Results

The following metrics were obtained from the notebook's held-out test set of 92 images.

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1-score (weighted) |
|---|---:|---:|---:|---:|
| **ResNet50** | **98.91%** | **98.95%** | **98.91%** | **98.90%** |
| DenseNet121 | 97.83% | 97.83% | 97.83% | 97.83% |

### Result summary

- ResNet50 correctly classified **91 out of 92** test images.
- DenseNet121 correctly classified **90 out of 92** test images.
- ResNet50 achieved the higher weighted F1-score in this experiment.
- Based on these test results, ResNet50 was the better-performing model for this dataset and run.

## Per-Class Test Performance

### ResNet50

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cat | 1.00 | 1.00 | 1.00 | 15 |
| Cow | 1.00 | 0.93 | 0.97 | 15 |
| Dog | 1.00 | 1.00 | 1.00 | 15 |
| Lamb | 0.97 | 1.00 | 0.98 | 32 |
| Zebra | 1.00 | 1.00 | 1.00 | 15 |

### DenseNet121

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Cat | 0.93 | 0.93 | 0.93 | 15 |
| Cow | 1.00 | 1.00 | 1.00 | 15 |
| Dog | 0.93 | 0.93 | 0.93 | 15 |
| Lamb | 1.00 | 1.00 | 1.00 | 32 |
| Zebra | 1.00 | 1.00 | 1.00 | 15 |

## Evaluation and Visualizations

The notebook includes:

- Training and validation accuracy curves
- Training and validation loss curves
- Classification reports
- Confusion matrices for both models
- Test metric comparison table
- Performance comparison chart
- A function to predict the class of a new image

## How to Run

1. Open the `.ipynb` notebook in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Connect Google Drive when prompted.
4. Place `dataset.zip` in Google Drive under `MyDrive` (or add a shortcut to it there).
5. Run the notebook cells in order.
6. Review the training logs, graphs, test metrics, and model comparison.

The notebook searches Google Drive for a file named `dataset.zip` and extracts it before loading the class folders.

## Saved Outputs

The notebook saves the trained model weights and evaluation outputs to:

```text
MyDrive/
└── ResNet50_DenseNet121_Results/
    ├── ResNet50_Final.pth
    ├── DenseNet121_Final.pth
    ├── model_comparison.csv
    └── class_names.txt
```

## Conclusion

In this experiment, both ResNet50 and DenseNet121 achieved strong results on the five-class image classification task. ResNet50 achieved **98.91% test accuracy** and a **98.90% weighted F1-score**, while DenseNet121 achieved **97.83%** for both metrics.

Therefore, ResNet50 performed slightly better on this particular dataset split. These results describe this single experiment and should not be interpreted as a guarantee of the same performance on other datasets or unseen data distributions.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- Google Drive

---

**Project:** Five-Class Image Classification  
**Architectures:** ResNet50 and DenseNet121  
**Task:** Multiclass Image Classification  
**Best model in this run:** ResNet50
