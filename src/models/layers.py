"""
Neural Network Layers Module

This module contains different types of layers used in neural networks.
Currently implements dense (fully connected) layers with various initialization methods.
"""

import numpy as np
from .activations import get_activation_function


class Layer:
    """Base class for neural network layers."""
    
    def forward(self, x):
        """Forward pass through the layer."""
        raise NotImplementedError
    
    def backward(self, grad_output):
        """Backward pass through the layer."""
        raise NotImplementedError
    
    def update_weights(self, learning_rate):
        """Update layer weights using gradients."""
        pass


class DenseLayer(Layer):
    """
    Fully connected (dense) layer.
    
    Performs linear transformation: output = input @ weights + bias
    Followed by optional activation function.
    """
    
    def __init__(self, input_size, output_size, activation='relu', 
                 weight_init='xavier', use_bias=True):
        """
        Initialize dense layer.
        
        Args:
            input_size (int): Number of input features
            output_size (int): Number of output features
            activation (str): Activation function name
            weight_init (str): Weight initialization method
            use_bias (bool): Whether to use bias terms
        """
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        
        # Initialize weights and biases
        self.weights = self._initialize_weights(weight_init)
        self.bias = np.zeros((1, output_size)) if use_bias else None
        
        # Get activation function
        self.activation = get_activation_function(activation)
        
        # Store for backward pass
        self.last_input = None
        self.last_linear_output = None
        
        # Gradients
        self.grad_weights = None
        self.grad_bias = None
        
        # For momentum optimizer
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_bias = np.zeros_like(self.bias) if use_bias else None
    
    def _initialize_weights(self, method):
        """
        Initialize weights using specified method.
        
        Args:
            method (str): Initialization method ('xavier', 'he', 'random', 'zeros')
            
        Returns:
            np.ndarray: Initialized weight matrix
        """
        if method == 'xavier':
            # Xavier/Glorot initialization - good for sigmoid/tanh
            limit = np.sqrt(6.0 / (self.input_size + self.output_size))
            return np.random.uniform(-limit, limit, (self.input_size, self.output_size))
        
        elif method == 'he':
            # He initialization - good for ReLU
            std = np.sqrt(2.0 / self.input_size)
            return np.random.normal(0, std, (self.input_size, self.output_size))
        
        elif method == 'random':
            # Small random values
            return np.random.randn(self.input_size, self.output_size) * 0.01
        
        elif method == 'zeros':
            # Zero initialization (not recommended for hidden layers)
            return np.zeros((self.input_size, self.output_size))
        
        else:
            raise ValueError(f"Unknown weight initialization method: {method}")
    
    def forward(self, x):
        """
        Forward pass through the dense layer.
        
        Args:
            x (np.ndarray): Input data of shape (batch_size, input_size)
            
        Returns:
            np.ndarray: Output after linear transformation and activation
        """
        # Store input for backward pass
        self.last_input = x
        
        # Linear transformation
        linear_output = np.dot(x, self.weights)
        if self.use_bias:
            linear_output += self.bias
        
        # Store linear output for backward pass
        self.last_linear_output = linear_output
        
        # Apply activation function
        activated_output = self.activation.forward(linear_output)
        
        return activated_output
    
    def backward(self, grad_output):
        """
        Backward pass through the dense layer.
        
        Args:
            grad_output (np.ndarray): Gradient from the next layer
            
        Returns:
            np.ndarray: Gradient with respect to the input
        """
        # Gradient of activation function
        grad_activation = self.activation.backward(self.last_linear_output)
        
        # Handle softmax case (returns Jacobian matrix)
        if hasattr(self.activation, '__class__') and \
           self.activation.__class__.__name__ == 'Softmax':
            # For softmax + cross-entropy, grad_output already includes the derivative
            grad_linear = grad_output
        else:
            # Element-wise multiplication for other activations
            grad_linear = grad_output * grad_activation
        
        # Gradients with respect to weights and bias
        self.grad_weights = np.dot(self.last_input.T, grad_linear)
        if self.use_bias:
            self.grad_bias = np.sum(grad_linear, axis=0, keepdims=True)
        
        # Gradient with respect to input
        grad_input = np.dot(grad_linear, self.weights.T)
        
        return grad_input
    
    def update_weights(self, learning_rate, momentum=0.0):
        """
        Update weights and biases using gradients.
        
        Args:
            learning_rate (float): Learning rate for weight updates
            momentum (float): Momentum coefficient for velocity updates
        """
        # Update weights with momentum
        self.velocity_weights = momentum * self.velocity_weights - learning_rate * self.grad_weights
        self.weights += self.velocity_weights
        
        # Update bias with momentum
        if self.use_bias:
            self.velocity_bias = momentum * self.velocity_bias - learning_rate * self.grad_bias
            self.bias += self.velocity_bias
    
    def get_params(self):
        """
        Get layer parameters.
        
        Returns:
            dict: Dictionary containing weights and bias
        """
        params = {'weights': self.weights.copy()}
        if self.use_bias:
            params['bias'] = self.bias.copy()
        return params
    
    def set_params(self, params):
        """
        Set layer parameters.
        
        Args:
            params (dict): Dictionary containing weights and bias
        """
        self.weights = params['weights'].copy()
        if self.use_bias and 'bias' in params:
            self.bias = params['bias'].copy()
    
    def __repr__(self):
        """String representation of the layer."""
        return (f"DenseLayer(input_size={self.input_size}, "
                f"output_size={self.output_size}, "
                f"activation={self.activation.__class__.__name__})")


class DropoutLayer(Layer):
    """
    Dropout layer for regularization.
    
    Randomly sets a fraction of input units to 0 during training.
    Helps prevent overfitting by reducing co-adaptation of neurons.
    """
    
    def __init__(self, dropout_rate=0.5):
        """
        Initialize dropout layer.
        
        Args:
            dropout_rate (float): Fraction of units to drop (0.0 to 1.0)
        """
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, x):
        """
        Forward pass through dropout layer.
        
        Args:
            x (np.ndarray): Input data
            
        Returns:
            np.ndarray: Output after applying dropout
        """
        if self.training:
            # Create random mask
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape)
            # Scale remaining values to maintain expected output
            return x * self.mask / (1 - self.dropout_rate)
        else:
            # During inference, return input as-is
            return x
    
    def backward(self, grad_output):
        """
        Backward pass through dropout layer.
        
        Args:
            grad_output (np.ndarray): Gradient from next layer
            
        Returns:
            np.ndarray: Gradient with dropout mask applied
        """
        if self.training:
            return grad_output * self.mask / (1 - self.dropout_rate)
        else:
            return grad_output
    
    def set_training(self, training):
        """
        Set training mode.
        
        Args:
            training (bool): Whether layer is in training mode
        """
        self.training = training
    
    def __repr__(self):
        """String representation of the layer."""
        return f"DropoutLayer(dropout_rate={self.dropout_rate})"
