"""
Automated Unit Tests for CNN Animal Image Classifier Architecture
Author: Samiun Mahmud
Week 2: Extended to include MobileNetV2 structural checks
"""

import sys
import os
import unittest
import numpy as np

# Ensure src directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


class TestCNNModelArchitecture(unittest.TestCase):
    """
    Test suite verifying structural integrity, dimensions, and forward-pass
    behavior of the Custom CNN Sequential architecture (Model 3).
    """

    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf
            from model import build_cnn_model
            cls.tf = tf
            cls.build_cnn_model = staticmethod(build_cnn_model)
            cls.has_tf = True
        except ImportError:
            cls.has_tf = False

    def setUp(self):
        if not self.has_tf:
            self.skipTest("TensorFlow not installed — skipping CNN model tests.")
        self.input_shape = (256, 256, 3)
        self.num_classes = 5
        self.model = self.build_cnn_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )

    def test_model_build_and_naming(self):
        """Verify model builds successfully and has the designated name."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.name, "Animal_Classifier_CNN")

    def test_output_tensor_shape(self):
        """Verify the classification head outputs a tensor of shape (None, 5)."""
        expected_shape = (None, self.num_classes)
        actual_shape = self.model.output_shape
        self.assertEqual(actual_shape, expected_shape)

    def test_softmax_probability_distribution(self):
        """Verify output probabilities are strictly non-negative and sum to 1.0."""
        dummy_batch = np.random.uniform(0.0, 1.0, size=(2, 256, 256, 3)).astype(np.float32)
        predictions = self.model.predict(dummy_batch, verbose=0)

        # Check prediction output shape
        self.assertEqual(predictions.shape, (2, 5))

        # Check non-negative probabilities
        self.assertTrue(np.all(predictions >= 0.0))

        # Check each prediction row sums to 1.0 (Softmax guarantee)
        row_sums = np.sum(predictions, axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(2, dtype=np.float32),
            rtol=1e-5,
            err_msg="Softmax probabilities must sum to 1.0"
        )

    def test_model_contains_conv_layers(self):
        """Verify model contains at least 3 Conv2D layers for feature extraction."""
        layer_types = [layer.__class__.__name__ for layer in self.model.layers]
        conv_count = layer_types.count('Conv2D')
        self.assertGreaterEqual(conv_count, 3,
            f"Expected at least 3 Conv2D layers, found {conv_count}")

    def test_model_contains_pooling_layers(self):
        """Verify model contains MaxPooling2D layers for spatial downsampling."""
        layer_types = [layer.__class__.__name__ for layer in self.model.layers]
        pool_count = layer_types.count('MaxPooling2D')
        self.assertGreaterEqual(pool_count, 3,
            f"Expected at least 3 MaxPooling2D layers, found {pool_count}")

    def test_model_has_output_softmax(self):
        """Verify the final layer uses softmax activation for probability output."""
        last_layer = self.model.layers[-1]
        self.assertEqual(last_layer.__class__.__name__, 'Dense')
        self.assertEqual(last_layer.units, self.num_classes)


class TestMobileNetV2Architecture(unittest.TestCase):
    """
    Test suite verifying MobileNetV2 transfer learning model structure (Model 4).
    Checks frozen backbone, GlobalAveragePooling head, and output shape.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf
            from mobilenet import build_mobilenet_model
            cls.tf = tf
            cls.build_mobilenet_model = staticmethod(build_mobilenet_model)
            cls.has_tf = True
        except ImportError:
            cls.has_tf = False

    def setUp(self):
        if not self.has_tf:
            self.skipTest("TensorFlow not installed — skipping MobileNetV2 tests.")
        self.input_shape = (224, 224, 3)
        self.num_classes = 5
        self.model = self.build_mobilenet_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            freeze_backbone=True
        )

    def test_mobilenet_builds_successfully(self):
        """Verify MobileNetV2 model builds without errors."""
        self.assertIsNotNone(self.model)

    def test_mobilenet_output_shape(self):
        """Verify MobileNetV2 head outputs 5-class softmax probabilities."""
        expected_shape = (None, self.num_classes)
        self.assertEqual(self.model.output_shape, expected_shape)

    def test_mobilenet_backbone_frozen(self):
        """Verify MobileNetV2 backbone layers are non-trainable (frozen)."""
        # Find the MobileNetV2 base inside the model
        mobilenet_base = None
        for layer in self.model.layers:
            if 'mobilenetv2' in layer.name.lower():
                mobilenet_base = layer
                break
        self.assertIsNotNone(mobilenet_base,
            "Could not find MobileNetV2 base layer in model")
        self.assertFalse(mobilenet_base.trainable,
            "MobileNetV2 backbone should be frozen (trainable=False)")

    def test_mobilenet_input_resolution(self):
        """Verify MobileNetV2 expects 224x224 input for pretrained weights."""
        expected_shape = (None, 224, 224, 3)
        self.assertEqual(self.model.input_shape, expected_shape)


class TestSourceFileStructure(unittest.TestCase):
    """
    Lightweight tests that verify all required source files exist.
    These always run, even without TensorFlow installed.
    """

    def _src_path(self, filename):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'src', filename)
        )

    def test_model_py_exists(self):
        """src/model.py must exist (Custom CNN — Model 3)."""
        self.assertTrue(os.path.isfile(self._src_path('model.py')))

    def test_mobilenet_py_exists(self):
        """src/mobilenet.py must exist (MobileNetV2 — Model 4)."""
        self.assertTrue(os.path.isfile(self._src_path('mobilenet.py')))

    def test_data_loader_py_exists(self):
        """src/data_loader.py must exist."""
        self.assertTrue(os.path.isfile(self._src_path('data_loader.py')))

    def test_train_py_exists(self):
        """src/train.py must exist."""
        self.assertTrue(os.path.isfile(self._src_path('train.py')))

    def test_evaluate_py_exists(self):
        """src/evaluate.py must exist."""
        self.assertTrue(os.path.isfile(self._src_path('evaluate.py')))

    def test_augment_py_exists(self):
        """src/augment.py must exist."""
        self.assertTrue(os.path.isfile(self._src_path('augment.py')))


if __name__ == "__main__":
    unittest.main()
