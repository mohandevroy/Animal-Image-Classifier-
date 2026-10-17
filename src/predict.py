"""
Single Image Inference Script for Animal Image Classification
Author: Samiun Mahmud
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from pathlib import Path

# Alphabetical classes mapping
CLASS_NAMES = ['cat', 'cow', 'dog', 'lamb', 'zebra']


def predict_image(img_path, model_path="models/cnn_animal_classifier.keras", class_names=None):
    """
    Performs inference on a single image file and returns prediction.

    Parameters:
        img_path (str or Path): Path to image file (.jpg, .jpeg, .png).
        model_path (str or Path): Path to saved Keras model file.
        class_names (list): List of class names ordered by index.

    Returns:
        tuple: (predicted_class, confidence_percentage)
    """
    if class_names is None:
        class_names = CLASS_NAMES

    path = Path(img_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at: {img_path}")

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    # Load model and predict
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(img_tensor, verbose=0)[0]

    top_idx = int(np.argmax(predictions))
    predicted_label = class_names[top_idx]
    confidence = float(predictions[top_idx]) * 100

    print("\n" + "=" * 45)
    print(f" Image       : {path.name}")
    print(f" Predicted   : {predicted_label.upper()}")
    print(f" Confidence  : {confidence:.2f}%")
    print("=" * 45)
    print(" All Class Probabilities:")
    for name, prob in zip(class_names, predictions):
        print(f"  • {name:<8}: {prob * 100:6.2f}%")
    print("=" * 45)

    return predicted_label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Animal Class for an Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--model", type=str, default="models/cnn_animal_classifier.keras", help="Path to trained model")
    args = parser.parse_args()

    predict_image(img_path=args.image, model_path=args.model)
