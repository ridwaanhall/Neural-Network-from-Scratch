"""
Activation Functions Module

This module contains various activation functions used in neural networks.
Each activation function includes both forward and backward (derivative) passes.
"""

import numpy as np


class ActivationFunction:
    """Base class for activation functions."""
    
    def forward(self, x):
        """Forward pass of the activation function."""
        raise NotImplementedError
    
    def backward(self, x):
        """Backward pass (derivative) of the activation function."""
        raise NotImplementedError


class ReLU(ActivationFunction):
    """
    Rectified Linear Unit (ReLU) activation function.
    
    ReLU(x) = max(0, x)
    
    Advantages:
    - Simple and fast computation
    - Helps mitigate vanishing gradient problem
    - Sparse activation (many neurons output 0)
    """
    
    def forward(self, x):
        """
        Forward pass of ReLU activation.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Output after applying ReLU
        """
        return np.maximum(0, x)
    
    def backward(self, x):
        """
        Backward pass of ReLU activation.
        
        Args:
            x (np.ndarray): Input array (same as forward pass input)
            
        Returns:
            np.ndarray: Derivative of ReLU
        """
        return (x > 0).astype(float)


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    
    Sigmoid(x) = 1 / (1 + exp(-x))
    
    Advantages:
    - Smooth gradient
    - Output bounded between 0 and 1
    
    Disadvantages:
    - Can cause vanishing gradient problem
    - Not zero-centered
    """
    
    def forward(self, x):
        """
        Forward pass of Sigmoid activation.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Output after applying Sigmoid
        """
        # Clip x to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def backward(self, x):
        """
        Backward pass of Sigmoid activation.
        
        Args:
            x (np.ndarray): Input array (same as forward pass input)
            
        Returns:
            np.ndarray: Derivative of Sigmoid
        """
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)


class Tanh(ActivationFunction):
    """
    Hyperbolic Tangent (Tanh) activation function.
    
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Advantages:
    - Zero-centered output
    - Smooth gradient
    
    Disadvantages:
    - Can cause vanishing gradient problem
    """
    
    def forward(self, x):
        """
        Forward pass of Tanh activation.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Output after applying Tanh
        """
        return np.tanh(x)
    
    def backward(self, x):
        """
        Backward pass of Tanh activation.
        
        Args:
            x (np.ndarray): Input array (same as forward pass input)
            
        Returns:
            np.ndarray: Derivative of Tanh
        """
        tanh_x = self.forward(x)
        return 1 - tanh_x**2


class Softmax(ActivationFunction):
    """
    Softmax activation function.
    
    Softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
    
    Used for multi-class classification output layer.
    Converts logits to probability distribution.
    """
    
    def forward(self, x):
        """
        Forward pass of Softmax activation.
        
        Args:
            x (np.ndarray): Input array of shape (batch_size, num_classes)
            
        Returns:
            np.ndarray: Probability distribution over classes
        """
        # Subtract max for numerical stability
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, x):
        """
        Backward pass of Softmax activation.
        
        Note: For softmax + cross-entropy loss, the gradient is computed
        differently in the loss function for efficiency.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Jacobian matrix of softmax
        """
        softmax_x = self.forward(x)
        # For each sample, compute Jacobian matrix
        batch_size, num_classes = softmax_x.shape
        jacobian = np.zeros((batch_size, num_classes, num_classes))
        
        for i in range(batch_size):
            s = softmax_x[i].reshape(-1, 1)
            jacobian[i] = np.diagflat(s) - np.dot(s, s.T)
        
        return jacobian


class LeakyReLU(ActivationFunction):
    """
    Leaky ReLU activation function.
    
    LeakyReLU(x) = x if x > 0, else alpha * x
    
    Advantages:
    - Allows small negative values to pass through
    - Helps prevent dying ReLU problem
    """
    
    def __init__(self, alpha=0.01):
        """
        Initialize Leaky ReLU with given slope for negative values.
        
        Args:
            alpha (float): Slope for negative values (default: 0.01)
        """
        self.alpha = alpha
    
    def forward(self, x):
        """
        Forward pass of Leaky ReLU activation.
        
        Args:
            x (np.ndarray): Input array
            
        Returns:
            np.ndarray: Output after applying Leaky ReLU
        """
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x):
        """
        Backward pass of Leaky ReLU activation.
        
        Args:
            x (np.ndarray): Input array (same as forward pass input)
            
        Returns:
            np.ndarray: Derivative of Leaky ReLU
        """
        return np.where(x > 0, 1, self.alpha)


# Factory function to get activation function by name
def get_activation_function(name):
    """
    Factory function to get activation function by name.
    
    Args:
        name (str): Name of the activation function
        
    Returns:
        ActivationFunction: Instance of the requested activation function
        
    Raises:
        ValueError: If activation function name is not recognized
    """
    activations = {
        'relu': ReLU(),
        'sigmoid': Sigmoid(),
        'tanh': Tanh(),
        'softmax': Softmax(),
        'leaky_relu': LeakyReLU()
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    
    return activations[name.lower()]
