"""
CNN Model Evaluation & Benchmarking Script
Author: Samiun Mahmud
"""

import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_and_split_data


def evaluate_model(model_path, data_dir, batch_size=32):
    """
    Evaluates a saved CNN model against the test dataset.
    Prints classification report and confusion matrix.
    """
    print(f"[Evaluation] Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    _, _, test_ds, class_names = load_and_split_data(
        data_dir=data_dir,
        batch_size=batch_size
    )

    y_true = []
    y_pred = []

    print("[Evaluation] Computing predictions on test split...")
    for X, y in test_ds:
        preds = model.predict(X, verbose=0)
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n" + "=" * 55)
    print("             CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    print("=" * 55)
    print("               CONFUSION MATRIX")
    print("=" * 55)
    print(cm)
    print("=" * 55)
    return cm, class_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN Model on Test Set")
    parser.add_argument("--model", type=str, default="models/cnn_animal_classifier.keras", help="Path to trained model")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root folder")
    args = parser.parse_args()

    evaluate_model(model_path=args.model, data_dir=args.data_dir)
