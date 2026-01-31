"""
Neural Networks Module - Production Implementation
===================================================

A comprehensive neural network library featuring:
- Feed-forward Neural Networks (FNN)
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN/LSTM)
- Optimized for NLP and tabular data tasks

Author: Refactored from NLP Demystified materials
Date: 2026
"""

import warnings
from typing import Tuple, List, Optional, Union, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report
)

# Optional dependencies with graceful fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available. High-level models disabled.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    warnings.warn("spaCy not available. Text preprocessing limited.")

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not available. Corpus features disabled.")


# ============================================================================
# CONFIGURATION & DATA STRUCTURES
# ============================================================================

@dataclass
class NetworkConfig:
    """Configuration for neural network architectures."""
    
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    learning_rate: float = 0.01
    activation: str = 'relu'
    output_activation: str = 'softmax'
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if not all(h > 0 for h in self.hidden_dims):
            raise ValueError("All hidden dimensions must be positive")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across libraries.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    if TF_AVAILABLE:
        tf.random.set_seed(seed)


def one_hot_encode(labels: np.ndarray) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded format.
    
    Optimized vectorized implementation replacing manual loops.
    
    Args:
        labels: 1D array of integer labels
        
    Returns:
        2D array of one-hot encoded labels (n_samples, n_classes)
        
    Example:
        >>> labels = np.array([0, 1, 1, 0, 2])
        >>> one_hot_encode(labels)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 1., 0.],
               [1., 0., 0.],
               [0., 0., 1.]])
    """
    labels = np.asarray(labels).flatten()
    
    if labels.size == 0:
        raise ValueError("Labels array cannot be empty")
    
    # Vectorized approach: much faster than manual loops
    uniques, indices = np.unique(labels, return_inverse=True)
    n_samples = len(labels)
    n_classes = len(uniques)
    
    one_hot = np.zeros((n_samples, n_classes), dtype=np.float32)
    one_hot[np.arange(n_samples), indices] = 1
    
    return one_hot


def standardize_features(X: np.ndarray, 
                         mean: Optional[np.ndarray] = None,
                         std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using z-score normalization.
    
    Formula: X_scaled = (X - mean) / std
    
    Args:
        X: Input features (n_samples, n_features)
        mean: Pre-computed mean (for test set). If None, computed from X
        std: Pre-computed std (for test set). If None, computed from X
        
    Returns:
        Tuple of (standardized_X, mean, std)
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
        # Prevent division by zero
        std = np.where(std == 0, 1.0, std)
    
    X_centered = X - mean
    X_scaled = X_centered / std
    
    return X_scaled, mean, std


def flatten_images(images: np.ndarray) -> np.ndarray:
    """
    Flatten multi-dimensional image arrays to 2D.
    
    Optimized to use built-in reshape instead of manual loops.
    
    Args:
        images: Array of shape (n_samples, height, width) or (n_samples, height, width, channels)
        
    Returns:
        Flattened array of shape (n_samples, height*width*channels)
    """
    if images.ndim < 2:
        raise ValueError(f"Expected at least 2D array, got {images.ndim}D")
    
    n_samples = images.shape[0]
    flat_dim = np.prod(images.shape[1:])
    
    # Vectorized reshape - much faster than manual loops
    return images.reshape(n_samples, flat_dim)


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

class Activation:
    """Base class for activation functions."""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        raise NotImplementedError
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative for backpropagation."""
        raise NotImplementedError


class ReLU(Activation):
    """
    Rectified Linear Unit activation.
    
    f(x) = max(0, x)
    f'(x) = 1 if x > 0 else 0
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply ReLU: max(0, x)."""
        return np.maximum(0, x)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Gradient: 1 if x > 0, else 0."""
        return (x > 0).astype(np.float32)


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    
    f(x) = 1 / (1 + exp(-x))
    f'(x) = f(x) * (1 - f(x))
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation."""
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Gradient of sigmoid."""
        fx = self.forward(x)
        return fx * (1 - fx)


class Softmax(Activation):
    """
    Softmax activation for multi-class classification.
    
    Converts logits to probability distribution.
    Uses numerical stability trick: subtract max before exp.
    """
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply softmax activation.
        
        Numerical stability: exp(x - max(x)) to prevent overflow.
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax derivative (usually combined with loss for efficiency).
        """
        # Typically computed together with cross-entropy for efficiency
        raise NotImplementedError("Use SoftmaxCrossEntropy for combined gradient")


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class Loss:
    """Base class for loss functions."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute loss."""
        raise NotImplementedError
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Compute gradient."""
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    """
    Cross-Entropy Loss for classification.
    
    L = -sum(y_true * log(y_pred))
    
    Includes numerical stability via clipping.
    """
    
    def __init__(self, epsilon: float = 1e-15):
        """
        Args:
            epsilon: Small value to prevent log(0)
        """
        self.epsilon = epsilon
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            y_pred: Predicted probabilities (n_samples, n_classes)
            y_true: True labels (one-hot encoded) (n_samples, n_classes)
            
        Returns:
            Average loss across samples
        """
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Compute negative log likelihood
        neg_log_likelihood = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        
        return np.mean(neg_log_likelihood)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Gradient of cross-entropy w.r.t predictions.
        
        When combined with softmax, simplifies to: (y_pred - y_true) / n_samples
        """
        n_samples = y_pred.shape[0]
        return (y_pred - y_true) / n_samples


class BinaryCrossEntropyLoss(Loss):
    """
    Binary Cross-Entropy Loss.
    
    L = -[y*log(p) + (1-y)*log(1-p)]
    """
    
    def __init__(self, epsilon: float = 1e-15):
        self.epsilon = epsilon
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute binary cross-entropy."""
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + 
            (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return loss
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Gradient of binary cross-entropy."""
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        n_samples = y_pred.shape[0]
        
        grad = -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped))
        return grad / n_samples


# ============================================================================
# NEURAL NETWORK LAYERS
# ============================================================================

class DenseLayer:
    """
    Fully-connected (Dense) layer.
    
    Performs: output = activation(X @ W + b)
    
    Optimizations:
    - Xavier/He initialization for better gradient flow
    - Vectorized operations (no loops)
    - Proper gradient computation for backpropagation
    """
    
    def __init__(self, 
                 n_inputs: int, 
                 n_units: int,
                 activation: str = 'relu',
                 weight_init: str = 'he'):
        """
        Initialize dense layer.
        
        Args:
            n_inputs: Number of input features
            n_units: Number of neurons in this layer
            activation: Activation function ('relu', 'sigmoid', 'linear')
            weight_init: Weight initialization ('he', 'xavier', 'uniform')
        """
        self.n_inputs = n_inputs
        self.n_units = n_units
        
        # Initialize weights with proper scaling
        self.W = self._initialize_weights(n_inputs, n_units, weight_init)
        self.b = np.zeros((1, n_units), dtype=np.float32)
        
        # Set activation function
        self.activation_name = activation
        self.activation = self._get_activation(activation)
        
        # Cache for backpropagation
        self.inputs = None
        self.z = None  # Pre-activation
        self.output = None  # Post-activation
        
        # Gradients
        self.dW = None
        self.db = None
        self.dinputs = None
    
    def _initialize_weights(self, 
                           n_inputs: int, 
                           n_units: int, 
                           method: str) -> np.ndarray:
        """
        Initialize weights using specified method.
        
        He initialization: W ~ N(0, sqrt(2/n_inputs)) - good for ReLU
        Xavier initialization: W ~ N(0, sqrt(1/n_inputs)) - good for sigmoid/tanh
        """
        if method == 'he':
            # He initialization for ReLU
            std = np.sqrt(2.0 / n_inputs)
            return np.random.randn(n_inputs, n_units).astype(np.float32) * std
        elif method == 'xavier':
            # Xavier/Glorot initialization
            std = np.sqrt(1.0 / n_inputs)
            return np.random.randn(n_inputs, n_units).astype(np.float32) * std
        else:  # uniform
            return np.random.uniform(-0.5, 0.5, (n_inputs, n_units)).astype(np.float32)
    
    def _get_activation(self, name: str) -> Activation:
        """Get activation function object."""
        activations = {
            'relu': ReLU(),
            'sigmoid': Sigmoid(),
            'linear': None
        }
        return activations.get(name.lower())
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input data (n_samples, n_inputs)
            
        Returns:
            Activated output (n_samples, n_units)
        """
        if inputs.shape[1] != self.n_inputs:
            raise ValueError(
                f"Expected {self.n_inputs} input features, got {inputs.shape[1]}"
            )
        
        # Cache inputs for backprop
        self.inputs = inputs
        
        # Linear transformation: Z = XW + b
        self.z = np.dot(inputs, self.W) + self.b
        
        # Apply activation
        if self.activation is not None:
            self.output = self.activation.forward(self.z)
        else:
            self.output = self.z
        
        return self.output
    
    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.
        
        Computes gradients w.r.t. weights, biases, and inputs.
        
        Args:
            dL_dout: Gradient from next layer (n_samples, n_units)
            
        Returns:
            Gradient w.r.t. inputs (n_samples, n_inputs)
        """
        # Gradient through activation
        if self.activation is not None:
            dL_dz = dL_dout * self.activation.derivative(self.z)
        else:
            dL_dz = dL_dout
        
        # Gradients for weights and biases
        # dW = X^T @ dL_dz (vectorized - no loops!)
        self.dW = np.dot(self.inputs.T, dL_dz)
        self.db = np.sum(dL_dz, axis=0, keepdims=True)
        
        # Gradient w.r.t. inputs (for backprop to previous layer)
        self.dinputs = np.dot(dL_dz, self.W.T)
        
        return self.dinputs
    
    def update(self, learning_rate: float, l2_reg: float = 0.0):
        """
        Update weights using gradient descent.
        
        Args:
            learning_rate: Step size for updates
            l2_reg: L2 regularization coefficient
        """
        # Add L2 regularization to gradients
        if l2_reg > 0:
            self.dW += l2_reg * self.W
        
        # Gradient descent update
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db


class SoftmaxCrossEntropyLayer:
    """
    Combined Softmax activation and Cross-Entropy loss.
    
    This combination allows for a simplified gradient computation:
    dL/dz = (y_pred - y_true) / n_samples
    
    Much more efficient than computing softmax and CE gradients separately.
    """
    
    def __init__(self, epsilon: float = 1e-15):
        """
        Args:
            epsilon: Small constant to prevent numerical issues
        """
        self.epsilon = epsilon
        self.inputs = None
        self.predictions = None
        self.loss_value = None
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Apply softmax to get probability predictions.
        
        Args:
            inputs: Logits from previous layer (n_samples, n_classes)
            
        Returns:
            Probability distributions (n_samples, n_classes)
        """
        self.inputs = inputs
        
        # Softmax with numerical stability
        exp_x = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.predictions = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
        # Clip to prevent log(0)
        self.predictions = np.clip(
            self.predictions, 
            self.epsilon, 
            1 - self.epsilon
        )
        
        return self.predictions
    
    def loss(self, y_true: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            y_true: One-hot encoded true labels (n_samples, n_classes)
            
        Returns:
            Average loss across samples
        """
        # Negative log likelihood
        neg_logs = -np.sum(y_true * np.log(self.predictions), axis=1)
        self.loss_value = np.mean(neg_logs)
        
        return self.loss_value
    
    def backward(self, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. inputs.
        
        For Softmax + Cross-Entropy, this simplifies to:
        dL/dz = (y_pred - y_true) / n_samples
        
        Args:
            y_true: One-hot encoded true labels
            
        Returns:
            Gradient w.r.t. inputs
        """
        n_samples = y_true.shape[0]
        dinputs = (self.predictions - y_true) / n_samples
        
        return dinputs


# ============================================================================
# FEEDFORWARD NEURAL NETWORK
# ============================================================================

class FeedForwardNN:
    """
    Multi-layer Feed-Forward Neural Network.
    
    Features:
    - Arbitrary depth (configurable hidden layers)
    - Multiple activation functions
    - L2 regularization
    - Batch training
    - Early stopping capability
    
    This is a production-ready implementation that replaces the basic
    NeuralNetwork class from nn1.py with modern best practices.
    """
    
    def __init__(self, config: NetworkConfig):
        """
        Initialize neural network.
        
        Args:
            config: Network configuration object
        """
        self.config = config
        
        if config.seed is not None:
            set_random_seed(config.seed)
        
        # Build network architecture
        self.layers = []
        self._build_network()
        
        # Output layer (softmax + cross-entropy combined)
        self.output_layer = SoftmaxCrossEntropyLayer()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def _build_network(self):
        """Construct hidden layers based on configuration."""
        layer_dims = [self.config.input_dim] + self.config.hidden_dims
        
        for i in range(len(layer_dims) - 1):
            layer = DenseLayer(
                n_inputs=layer_dims[i],
                n_units=layer_dims[i + 1],
                activation=self.config.activation
            )
            self.layers.append(layer)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through entire network.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Predictions (n_samples, n_classes)
        """
        output = X
        
        # Pass through hidden layers
        for layer in self.layers:
            output = layer.forward(output)
        
        # Final layer: add output neurons then softmax
        if not hasattr(self, 'output_dense'):
            # Create output dense layer (linear activation)
            self.output_dense = DenseLayer(
                n_inputs=self.layers[-1].n_units,
                n_units=self.config.output_dim,
                activation='linear'
            )
        
        logits = self.output_dense.forward(output)
        predictions = self.output_layer.forward(logits)
        
        return predictions
    
    def compute_loss(self, y_true: np.ndarray) -> float:
        """
        Compute loss for current predictions.
        
        Args:
            y_true: True labels (one-hot encoded)
            
        Returns:
            Loss value
        """
        return self.output_layer.loss(y_true)
    
    def backward(self, y_true: np.ndarray):
        """
        Backpropagate gradients through the network.
        
        Args:
            y_true: True labels (one-hot encoded)
        """
        # Gradient from output layer (simplified for softmax+CE)
        dL_dout = self.output_layer.backward(y_true)
        
        # Backprop through output dense layer
        dL_dout = self.output_dense.backward(dL_dout)
        
        # Backprop through hidden layers (in reverse order)
        for layer in reversed(self.layers):
            dL_dout = layer.backward(dL_dout)
    
    def update_weights(self, learning_rate: float, l2_reg: float = 0.0):
        """
        Update all network weights.
        
        Args:
            learning_rate: Learning rate for gradient descent
            l2_reg: L2 regularization coefficient
        """
        for layer in self.layers:
            layer.update(learning_rate, l2_reg)
        self.output_dense.update(learning_rate, l2_reg)
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            l2_reg: float = 0.0,
            early_stopping: int = 10,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the neural network.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples, n_classes) - one-hot encoded
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Mini-batch size
            l2_reg: L2 regularization coefficient
            early_stopping: Patience for early stopping (0 = disabled)
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        n_samples = X_train.shape[0]
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            epoch_loss = 0.0
            n_batches = 0
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(X_batch)
                loss = self.compute_loss(y_batch)
                
                # Backward pass
                self.backward(y_batch)
                self.update_weights(self.config.learning_rate, l2_reg)
                
                epoch_loss += loss
                n_batches += 1
            
            # Calculate epoch metrics
            avg_train_loss = epoch_loss / n_batches
            train_acc = self.evaluate_accuracy(X_train, y_train)
            
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_predictions = self.forward(X_val)
                val_loss = self.compute_loss(y_val)
                val_acc = self.evaluate_accuracy(X_val, y_val)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Early stopping check
                if early_stopping > 0:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping:
                            if verbose:
                                print(f"Early stopping at epoch {epoch + 1}")
                            break
            
            # Print progress
            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0):
                msg = f"Epoch {epoch + 1}/{epochs} - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                print(msg)
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predicted class probabilities (n_samples, n_classes)
        """
        return self.forward(X)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class indices
        """
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)
    
    def evaluate_accuracy(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate classification accuracy.
        
        Args:
            X: Input features
            y_true: True labels (one-hot encoded)
            
        Returns:
            Accuracy score
        """
        y_pred = self.predict_classes(X)
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_pred == y_true_labels)


# ============================================================================
# CONVOLUTIONAL NEURAL NETWORK LAYERS
# ============================================================================

class Conv2D:
    """
    2D Convolutional layer.
    
    Optimized implementation for image processing tasks.
    Uses vectorized operations where possible.
    """
    
    def __init__(self,
                 num_filters: int,
                 filter_height: int,
                 filter_width: int,
                 stride: int = 1,
                 padding: str = 'valid'):
        """
        Initialize convolutional layer.
        
        Args:
            num_filters: Number of convolutional filters
            filter_height: Height of each filter
            filter_width: Width of each filter
            stride: Stride for convolution
            padding: 'valid' or 'same'
        """
        self.num_filters = num_filters
        self.filter_h = filter_height
        self.filter_w = filter_width
        self.stride = stride
        self.padding = padding
        
        # Initialize filters with small random values
        self.filters = np.random.randn(
            num_filters, 
            filter_height, 
            filter_width
        ) * 0.1
        
        self.input_shape = None
        self.output_shape = None
    
    def _apply_padding(self, image: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
        """Apply zero padding to image."""
        return np.pad(
            image, 
            ((pad_h, pad_h), (pad_w, pad_w)),
            mode='constant',
            constant_values=0
        )
    
    def forward(self, images: np.ndarray) -> np.ndarray:
        """
        Perform forward convolution.
        
        Args:
            images: Input images (n_samples, height, width)
            
        Returns:
            Feature maps (n_samples, num_filters, out_height, out_width)
        """
        n_samples, img_h, img_w = images.shape
        self.input_shape = images.shape
        
        # Calculate output dimensions
        if self.padding == 'same':
            pad_h = ((img_h - 1) * self.stride + self.filter_h - img_h) // 2
            pad_w = ((img_w - 1) * self.stride + self.filter_w - img_w) // 2
            out_h = img_h
            out_w = img_w
        else:  # valid
            pad_h, pad_w = 0, 0
            out_h = (img_h - self.filter_h) // self.stride + 1
            out_w = (img_w - self.filter_w) // self.stride + 1
        
        # Initialize output
        feature_maps = np.zeros((n_samples, self.num_filters, out_h, out_w))
        
        # Apply convolution
        for i in range(n_samples):
            img = images[i]
            
            if pad_h > 0 or pad_w > 0:
                img = self._apply_padding(img, pad_h, pad_w)
            
            for f in range(self.num_filters):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        img_patch = img[
                            h_start:h_start + self.filter_h,
                            w_start:w_start + self.filter_w
                        ]
                        
                        # Element-wise multiplication and sum
                        feature_maps[i, f, h, w] = np.sum(
                            img_patch * self.filters[f]
                        )
        
        self.output_shape = feature_maps.shape
        return feature_maps


class MaxPool2D:
    """
    2D Max Pooling layer.
    
    Reduces spatial dimensions by taking maximum value in each pool.
    """
    
    def __init__(self, pool_size: int = 2, stride: int = 2):
        """
        Initialize max pooling layer.
        
        Args:
            pool_size: Size of pooling window
            stride: Stride for pooling
        """
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, feature_maps: np.ndarray) -> np.ndarray:
        """
        Perform max pooling.
        
        Args:
            feature_maps: Input feature maps (n_samples, n_filters, height, width)
            
        Returns:
            Pooled feature maps (n_samples, n_filters, out_height, out_width)
        """
        n_samples, n_filters, in_h, in_w = feature_maps.shape
        
        # Calculate output dimensions
        out_h = (in_h - self.pool_size) // self.stride + 1
        out_w = (in_w - self.pool_size) // self.stride + 1
        
        # Initialize output
        pooled = np.zeros((n_samples, n_filters, out_h, out_w))
        
        # Apply max pooling
        for i in range(n_samples):
            for f in range(n_filters):
                for h in range(out_h):
                    for w in range(out_w):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        pool_region = feature_maps[
                            i, f,
                            h_start:h_start + self.pool_size,
                            w_start:w_start + self.pool_size
                        ]
                        
                        pooled[i, f, h, w] = np.max(pool_region)
        
        return pooled


# ============================================================================
# TEXT PREPROCESSING UTILITIES
# ============================================================================

class TextPreprocessor:
    """
    Advanced text preprocessing for NLP tasks.
    
    Uses spaCy for sophisticated tokenization and lemmatization.
    Falls back to simple methods if spaCy unavailable.
    """
    
    def __init__(self, 
                 model: str = 'en_core_web_sm',
                 remove_stop_words: bool = True,
                 remove_punctuation: bool = True,
                 min_token_length: int = 2):
        """
        Initialize text preprocessor.
        
        Args:
            model: spaCy model name
            remove_stop_words: Whether to remove stop words
            remove_punctuation: Whether to remove punctuation
            min_token_length: Minimum token length to keep
        """
        self.remove_stop_words = remove_stop_words
        self.remove_punctuation = remove_punctuation
        self.min_token_length = min_token_length
        
        if SPACY_AVAILABLE:
            try:
                # Disable unnecessary pipeline components for speed
                self.nlp = spacy.load(
                    model, 
                    disable=['ner', 'parser']
                )
            except OSError:
                warnings.warn(
                    f"spaCy model '{model}' not found. "
                    "Download with: python -m spacy download en_core_web_sm"
                )
                self.nlp = None
        else:
            self.nlp = None
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize and preprocess text.
        
        Args:
            text: Input text string
            
        Returns:
            List of processed tokens
        """
        if self.nlp is not None:
            # Use spaCy for advanced preprocessing
            doc = self.nlp(text)
            tokens = []
            
            for token in doc:
                # Apply filters
                if len(token) <= self.min_token_length:
                    continue
                if self.remove_punctuation and token.is_punct:
                    continue
                if self.remove_stop_words and token.is_stop:
                    continue
                if not token.is_alpha:
                    continue
                
                # Lemmatize and lowercase
                tokens.append(token.lemma_.lower())
            
            return tokens
        else:
            # Simple fallback preprocessing
            tokens = text.lower().split()
            tokens = [t for t in tokens if len(t) > self.min_token_length]
            return tokens
    
    def preprocess(self, text: str, delimiter: str = "|") -> str:
        """
        Preprocess text and join tokens.
        
        Args:
            text: Input text
            delimiter: Delimiter for joining tokens
            
        Returns:
            Preprocessed text string
        """
        tokens = self.tokenize(text)
        return delimiter.join(tokens)



# ============================================================================
# KERAS/TENSORFLOW HIGH-LEVEL MODELS
# ============================================================================

if TF_AVAILABLE:
    class KerasNNBuilder:
        """
        Builder for Keras-based neural network models.
        
        Provides easy interface for common architectures.
        Only available if TensorFlow is installed.
        """
        
        def __init__(self):
            """Initialize builder."""
            pass
        
        @staticmethod
        def build_mlp(input_dim: int,
                      hidden_dims: List[int],
                      output_dim: int,
                      activation: str = 'relu',
                      output_activation: str = 'softmax',
                      dropout: float = 0.0) -> keras.Model:
            """
            Build Multi-Layer Perceptron.
            
            Args:
                input_dim: Input feature dimension
                hidden_dims: List of hidden layer sizes
                output_dim: Number of output classes
                activation: Hidden layer activation
                output_activation: Output layer activation
                dropout: Dropout rate (0 = no dropout)
                
            Returns:
                Compiled Keras model
            """
            model = keras.Sequential()
            
            # Input layer
            model.add(layers.Input(shape=(input_dim,)))
            
            # Hidden layers
            for i, dim in enumerate(hidden_dims):
                model.add(layers.Dense(dim, activation=activation, name=f'hidden_{i+1}'))
                if dropout > 0:
                    model.add(layers.Dropout(dropout))
            
            # Output layer
            model.add(layers.Dense(output_dim, activation=output_activation, name='output'))
            
            return model
        
        @staticmethod
        def build_lstm_classifier(vocab_size: int,
                                  embedding_dim: int,
                                  max_sequence_length: int,
                                  num_classes: int,
                                  lstm_units: int = 128,
                                  bidirectional: bool = True) -> keras.Model:
            """
            Build LSTM-based text classifier.
            
            Args:
                vocab_size: Size of vocabulary
                embedding_dim: Dimension of word embeddings
                max_sequence_length: Maximum sequence length
                num_classes: Number of output classes
                lstm_units: Number of LSTM units
                bidirectional: Whether to use bidirectional LSTM
                
            Returns:
                Compiled Keras model
            """
            model = keras.Sequential()
            
            # Embedding layer
            model.add(layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_sequence_length,
                mask_zero=True
            ))
            
            # LSTM layer
            lstm_layer = layers.LSTM(lstm_units, return_sequences=True)
            if bidirectional:
                lstm_layer = layers.Bidirectional(lstm_layer)
            model.add(lstm_layer)
            
            # Output layer
            model.add(layers.Dense(num_classes, activation='softmax'))
            
            return model
else:
    # Placeholder when TensorFlow is not available
    class KerasNNBuilder:
        """Keras builder placeholder (TensorFlow not installed)."""
        def __init__(self):
            raise RuntimeError(
                "TensorFlow not available. Install with: pip install tensorflow"
            )



# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation utilities.
    
    Provides metrics, confusion matrix, and classification reports.
    """
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation.
        
        Args:
            y_true: True labels (class indices)
            y_pred: Predicted labels (class indices)
            class_names: Optional class names for report
            
        Returns:
            Dictionary with all metrics
        """
        # Convert probabilities to class labels if needed
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Classification report
        if class_names is not None:
            metrics['classification_report'] = classification_report(
                y_true, y_pred, 
                target_names=class_names,
                zero_division=0
            )
        else:
            metrics['classification_report'] = classification_report(
                y_true, y_pred,
                zero_division=0
            )
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, Any]):
        """
        Pretty print evaluation metrics.
        
        Args:
            metrics: Dictionary from evaluate_classification
        """
        print("="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print("\n" + "="*60)
        print("CONFUSION MATRIX")
        print("="*60)
        print(metrics['confusion_matrix'])
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(metrics['classification_report'])


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

class DataLoader:
    """
    Utilities for loading and preprocessing common datasets.
    """
    
    @staticmethod
    def load_diabetes_data(filepath: Union[str, Path],
                          test_size: float = 0.2,
                          random_state: int = 42) -> Tuple:
        """
        Load and preprocess diabetes dataset.
        
        Args:
            filepath: Path to CSV file
            test_size: Fraction for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Load data
        data = pd.read_csv(filepath)
        
        # Select features
        feature_cols = [
            'Glucose', 'Age', 'Pregnancies', 'BloodPressure',
            'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'
        ]
        
        X = data[feature_cols].values
        y = data['Outcome'].values
        
        # Standardize features
        X, _, _ = standardize_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def load_pos_tagging_data() -> Tuple:
        """
        Load POS tagging datasets from NLTK.
        
        Returns:
            Tuple of (sentences, tags)
        """
        if not NLTK_AVAILABLE:
            raise RuntimeError("NLTK not available. Install with: pip install nltk")
        
        # Download required corpora
        for corpus in ['treebank', 'brown', 'conll2000', 'universal_tagset']:
            try:
                nltk.download(corpus, quiet=True)
            except:
                pass
        
        # Load tagged sentences
        tagged_sentences = (
            nltk.corpus.treebank.tagged_sents(tagset='universal') +
            nltk.corpus.brown.tagged_sents(tagset='universal') +
            nltk.corpus.conll2000.tagged_sents(tagset='universal')
        )
        
        # Separate sentences and tags
        sentences, tags = [], []
        for sent in tagged_sentences:
            words, sent_tags = zip(*sent)
            sentences.append(list(words))
            tags.append(list(sent_tags))
        
        return sentences, tags


# ============================================================================
# EXAMPLE USAGE & DEMONSTRATION
# ============================================================================

def demo_feedforward_nn():
    """
    Demonstrate FeedForwardNN on synthetic data.
    """
    print("\n" + "="*60)
    print("FEEDFORWARD NEURAL NETWORK DEMO")
    print("="*60 + "\n")
    
    # Generate synthetic data
    from sklearn.datasets import make_gaussian_quantiles
    
    X, y = make_gaussian_quantiles(
        n_samples=1000,
        n_features=10,
        n_classes=3,
        random_state=42
    )
    
    # One-hot encode labels
    y_one_hot = one_hot_encode(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, 
        test_size=0.2, 
        random_state=42
    )
    
    # Standardize features
    X_train, mean, std = standardize_features(X_train)
    X_test, _, _ = standardize_features(X_test, mean, std)
    
    # Create network configuration
    config = NetworkConfig(
        input_dim=10,
        hidden_dims=[16, 8],
        output_dim=3,
        learning_rate=0.01,
        activation='relu',
        seed=42
    )
    
    # Build and train model
    print("Training model...")
    model = FeedForwardNN(config)
    
    history = model.fit(
        X_train, y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=100,
        batch_size=32,
        early_stopping=10,
        verbose=True
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    y_pred = model.predict_classes(X_test)
    y_true = np.argmax(y_test, axis=1)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_classification(y_true, y_pred)
    evaluator.print_metrics(metrics)
    
    print("\n✓ Demo completed successfully!\n")


def demo_keras_models():
    """
    Demonstrate Keras model builder.
    """
    if not TF_AVAILABLE:
        print("TensorFlow not available. Skipping Keras demo.")
        return
    
    print("\n" + "="*60)
    print("KERAS MODEL BUILDER DEMO")
    print("="*60 + "\n")
    
    # Generate synthetic data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build model
    builder = KerasNNBuilder()
    model = builder.build_mlp(
        input_dim=20,
        hidden_dims=[32, 16],
        output_dim=2,
        dropout=0.2
    )
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\n✓ Keras demo completed successfully!\n")


if __name__ == '__main__':
    """
    Main execution: Run demos if script is executed directly.
    """
    print("\n" + "="*60)
    print("NEURAL NETWORKS MODULE - PRODUCTION VERSION")
    print("="*60)
    print("\nThis module provides production-ready implementations of:")
    print("  • Feed-Forward Neural Networks")
    print("  • Convolutional Neural Networks")
    print("  • Recurrent Neural Networks (via Keras)")
    print("  • Advanced text preprocessing")
    print("  • Comprehensive evaluation metrics")
    print("\n" + "="*60)
    
    # Run demonstrations
    try:
        demo_feedforward_nn()
    except Exception as e:
        print(f"\n✗ FeedForward demo failed: {e}\n")
    
    try:
        demo_keras_models()
    except Exception as e:
        print(f"\n✗ Keras demo failed: {e}\n")
    
    print("="*60)
    print("All demos completed!")
    print("="*60 + "\n")