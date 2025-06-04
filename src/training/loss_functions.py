"""
Loss Functions Module

This module contains various loss functions used for training neural networks.
Each loss function includes both forward pass (computing loss) and backward pass (computing gradients).
"""

import numpy as np


class LossFunction:
    """Base class for loss functions."""
    
    def forward(self, predictions, targets):
        """Compute the loss."""
        raise NotImplementedError
    
    def backward(self, predictions, targets):
        """Compute gradients with respect to predictions."""
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean Squared Error (MSE) loss function.
    
    MSE = (1/n) * sum((predictions - targets)^2)
    
    Commonly used for regression tasks.
    """
    
    def forward(self, predictions, targets):
        """
        Compute MSE loss.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): True target values
            
        Returns:
            float: MSE loss value
        """
        diff = predictions - targets
        return np.mean(diff ** 2)
    
    def backward(self, predictions, targets):
        """
        Compute gradient of MSE loss.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): True target values
            
        Returns:
            np.ndarray: Gradient with respect to predictions
        """
        n = predictions.shape[0]
        return 2 * (predictions - targets) / n


class CrossEntropyLoss(LossFunction):
    """
    Cross-Entropy loss function.
    
    Used for multi-class classification with softmax output.
    
    CrossEntropy = -sum(targets * log(predictions))
    
    Note: Assumes predictions are already softmax probabilities
    and targets are one-hot encoded.
    """
    
    def forward(self, predictions, targets):
        """
        Compute cross-entropy loss.
        
        Args:
            predictions (np.ndarray): Softmax probabilities of shape (batch_size, num_classes)
            targets (np.ndarray): One-hot encoded targets of shape (batch_size, num_classes)
            
        Returns:
            float: Cross-entropy loss value
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Compute cross-entropy loss
        loss = -np.sum(targets * np.log(predictions)) / predictions.shape[0]
        return loss
    
    def backward(self, predictions, targets):
        """
        Compute gradient of cross-entropy loss.
        
        For softmax + cross-entropy combination, the gradient is simply:
        gradient = predictions - targets
        
        Args:
            predictions (np.ndarray): Softmax probabilities
            targets (np.ndarray): One-hot encoded targets
            
        Returns:
            np.ndarray: Gradient with respect to predictions
        """
        return (predictions - targets) / predictions.shape[0]


class BinaryCrossEntropyLoss(LossFunction):
    """
    Binary Cross-Entropy loss function.
    
    Used for binary classification tasks.
    
    BCE = -[y*log(p) + (1-y)*log(1-p)]
    
    where y is the true label (0 or 1) and p is the predicted probability.
    """
    
    def forward(self, predictions, targets):
        """
        Compute binary cross-entropy loss.
        
        Args:
            predictions (np.ndarray): Predicted probabilities
            targets (np.ndarray): True binary labels (0 or 1)
            
        Returns:
            float: Binary cross-entropy loss value
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Compute binary cross-entropy loss
        loss = -(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        return np.mean(loss)
    
    def backward(self, predictions, targets):
        """
        Compute gradient of binary cross-entropy loss.
        
        Args:
            predictions (np.ndarray): Predicted probabilities
            targets (np.ndarray): True binary labels
            
        Returns:
            np.ndarray: Gradient with respect to predictions
        """
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        grad = -(targets / predictions - (1 - targets) / (1 - predictions))
        return grad / predictions.shape[0]


class CategoricalCrossEntropyLoss(LossFunction):
    """
    Categorical Cross-Entropy loss with label smoothing option.
    
    Enhanced version of cross-entropy loss with optional label smoothing
    for better generalization.
    """
    
    def __init__(self, label_smoothing=0.0):
        """
        Initialize categorical cross-entropy loss.
        
        Args:
            label_smoothing (float): Label smoothing factor (0.0 to 1.0)
        """
        self.label_smoothing = label_smoothing
    
    def _smooth_labels(self, targets, num_classes):
        """
        Apply label smoothing to targets.
        
        Args:
            targets (np.ndarray): One-hot encoded targets
            num_classes (int): Number of classes
            
        Returns:
            np.ndarray: Smoothed targets
        """
        if self.label_smoothing == 0.0:
            return targets
        
        smoothed = targets * (1 - self.label_smoothing)
        smoothed += self.label_smoothing / num_classes
        return smoothed
    
    def forward(self, predictions, targets):
        """
        Compute categorical cross-entropy loss with optional label smoothing.
        
        Args:
            predictions (np.ndarray): Softmax probabilities
            targets (np.ndarray): One-hot encoded targets
            
        Returns:
            float: Categorical cross-entropy loss value
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0.0:
            num_classes = targets.shape[1]
            targets = self._smooth_labels(targets, num_classes)
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Compute cross-entropy loss
        loss = -np.sum(targets * np.log(predictions)) / predictions.shape[0]
        return loss
    
    def backward(self, predictions, targets):
        """
        Compute gradient of categorical cross-entropy loss.
        
        Args:
            predictions (np.ndarray): Softmax probabilities
            targets (np.ndarray): One-hot encoded targets
            
        Returns:
            np.ndarray: Gradient with respect to predictions
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0.0:
            num_classes = targets.shape[1]
            targets = self._smooth_labels(targets, num_classes)
        
        return (predictions - targets) / predictions.shape[0]


class HuberLoss(LossFunction):
    """
    Huber loss function.
    
    Combines MSE and MAE losses. Less sensitive to outliers than MSE.
    Uses MSE for small errors and MAE for large errors.
    """
    
    def __init__(self, delta=1.0):
        """
        Initialize Huber loss.
        
        Args:
            delta (float): Threshold for switching between MSE and MAE
        """
        self.delta = delta
    
    def forward(self, predictions, targets):
        """
        Compute Huber loss.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): True target values
            
        Returns:
            float: Huber loss value
        """
        diff = predictions - targets
        abs_diff = np.abs(diff)
        
        # Use MSE for small errors, MAE for large errors
        quadratic = 0.5 * diff ** 2
        linear = self.delta * abs_diff - 0.5 * self.delta ** 2
        
        loss = np.where(abs_diff <= self.delta, quadratic, linear)
        return np.mean(loss)
    
    def backward(self, predictions, targets):
        """
        Compute gradient of Huber loss.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): True target values
            
        Returns:
            np.ndarray: Gradient with respect to predictions
        """
        diff = predictions - targets
        abs_diff = np.abs(diff)
        
        # Gradient is diff for small errors, delta*sign(diff) for large errors
        grad = np.where(abs_diff <= self.delta, diff, self.delta * np.sign(diff))
        return grad / predictions.shape[0]


# Factory function to get loss function by name
def get_loss_function(name, **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        name (str): Name of the loss function
        **kwargs: Additional arguments for loss function initialization
        
    Returns:
        LossFunction: Instance of the requested loss function
        
    Raises:
        ValueError: If loss function name is not recognized
    """
    loss_functions = {
        'mse': MeanSquaredError,
        'mean_squared_error': MeanSquaredError,
        'cross_entropy': CrossEntropyLoss,
        'categorical_crossentropy': CrossEntropyLoss,
        'binary_crossentropy': BinaryCrossEntropyLoss,
        'categorical_crossentropy_smooth': CategoricalCrossEntropyLoss,
        'huber': HuberLoss
    }
    
    if name.lower() not in loss_functions:
        raise ValueError(f"Unknown loss function: {name}")
    
    return loss_functions[name.lower()](**kwargs)
