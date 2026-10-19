"""
Automated Unit Tests for CNN Animal Image Classifier Architecture
Author: Samiun Mahmud
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
    behavior of the Convolutional Neural Network Sequential architecture.
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
            self.skipTest("TensorFlow is not installed in the local environment.")
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

    def test_input_tensor_shape(self):
        """Verify the input layer expects a tensor of shape (None, 256, 256, 3)."""
        expected_shape = (None, 256, 256, 3)
        actual_shape = self.model.layers[0].input_shape
        self.assertEqual(actual_shape, expected_shape)

    def test_output_tensor_shape(self):
        """Verify the classification head outputs a tensor of shape (None, 5)."""
        expected_shape = (None, self.num_classes)
        actual_shape = self.model.output_shape
        self.assertEqual(actual_shape, expected_shape)

    def test_softmax_probability_distribution(self):
        """Verify output probabilities are strictly non-negative and sum to 1.0."""
        # Generate a dummy batch of 2 normalized images
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

    def test_layer_hierarchy(self):
        """Verify layer sequence: 3 Conv blocks, Flatten, and 2 Dense layers."""
        layer_types = [layer.__class__.__name__ for layer in self.model.layers]
        expected_types = [
            'Conv2D',
            'MaxPooling2D',
            'Conv2D',
            'MaxPooling2D',
            'Conv2D',
            'MaxPooling2D',
            'Flatten',
            'Dense',
            'Dense'
        ]
        self.assertEqual(layer_types, expected_types)


if __name__ == "__main__":
    unittest.main()
