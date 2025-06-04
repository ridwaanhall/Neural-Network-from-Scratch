"""
Neural Network Model

This module contains the main neural network class that combines layers,
loss functions, and training algorithms into a complete model.
"""

import numpy as np
import pickle
import json
from datetime import datetime
from .layers import DenseLayer, DropoutLayer
from ..training.loss_functions import get_loss_function
from ..utils.metrics import accuracy_score, confusion_matrix


class NeuralNetwork:
    """
    A feedforward neural network implementation from scratch.
    
    This class provides a complete neural network with customizable architecture,
    training algorithms, and evaluation metrics.
    """
    def __init__(self, input_size=None, hidden_layers=None, output_size=None, 
                 activation='relu', output_activation='softmax',
                 loss='cross_entropy', learning_rate=0.001,
                 weight_init='xavier', use_dropout=False, dropout_rate=0.5):
        """
        Initialize the neural network.
        
        Can be used in two ways:
        1. Constructor-based: Pass all parameters to build the network immediately
        2. Manual building: Create empty network and use add_layer() method
        
        Args:
            input_size (int, optional): Number of input features
            hidden_layers (list, optional): List of hidden layer sizes
            output_size (int, optional): Number of output classes
            activation (str): Activation function for hidden layers
            output_activation (str): Activation function for output layer
            loss (str): Loss function name
            learning_rate (float): Learning rate for optimization
            weight_init (str): Weight initialization method
            use_dropout (bool): Whether to use dropout regularization
            dropout_rate (float): Dropout rate if dropout is enabled
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers or []
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.output_activation = output_activation
        self.weight_init = weight_init
        
        # Initialize layers list
        self.layers = []
        
        # Build network if parameters provided
        if input_size is not None and output_size is not None:
            self._build_network(activation, output_activation, weight_init)
        
        # Initialize loss function
        self.loss_function = get_loss_function(loss)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Training state
        self.is_training = True
    
    def _build_network(self, activation, output_activation, weight_init):
        """
        Build the neural network architecture.
        
        Args:
            activation (str): Activation function for hidden layers
            output_activation (str): Activation function for output layer
            weight_init (str): Weight initialization method
        """
        # Input layer to first hidden layer
        prev_size = self.input_size
        
        # Add hidden layers
        for i, layer_size in enumerate(self.hidden_layers):
            # Add dense layer
            layer = DenseLayer(
                input_size=prev_size,
                output_size=layer_size,
                activation=activation,
                weight_init=weight_init
            )
            self.layers.append(layer)
            
            # Add dropout layer if enabled
            if self.use_dropout and i < len(self.hidden_layers) - 1:  # Not after last hidden layer
                dropout_layer = DropoutLayer(dropout_rate=self.dropout_rate)
                self.layers.append(dropout_layer)
            
            prev_size = layer_size
          # Add output layer
        output_layer = DenseLayer(
            input_size=prev_size,
            output_size=self.output_size,
            activation=output_activation,
            weight_init=weight_init
        )
        self.layers.append(output_layer)
    
    def add_layer(self, layer):
        """
        Add a layer to the network.
        
        This method allows building networks layer by layer for more flexibility.
        
        Args:
            layer: A layer object (DenseLayer, DropoutLayer, etc.)
        """
        self.layers.append(layer)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (np.ndarray): Input data of shape (batch_size, input_size)
            
        Returns:
            np.ndarray: Network output of shape (batch_size, output_size)
        """
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad_output):
        """
        Backward pass through the network.
        
        Args:
            grad_output (np.ndarray): Gradient from loss function
            
        Returns:
            np.ndarray: Gradient with respect to input
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    def update_weights(self, momentum=0.9):
        """
        Update network weights using computed gradients.
        
        Args:
            momentum (float): Momentum coefficient for optimization
        """
        for layer in self.layers:
            # Only update weights for layers that have trainable parameters
            if hasattr(layer, 'weights') and hasattr(layer, 'update_weights'):
                layer.update_weights(self.learning_rate, momentum)
    
    def set_training(self, training):
        """
        Set training mode for all layers.
        
        Args:
            training (bool): Whether network is in training mode
        """
        self.is_training = training
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(training)
    
    def predict(self, x):
        """
        Make predictions on input data.
        
        Args:
            x (np.ndarray): Input data
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        self.set_training(False)
        predictions = self.forward(x)
        self.set_training(True)
        return predictions
    
    def predict_classes(self, x):
        """
        Predict class labels for input data.
        
        Args:
            x (np.ndarray): Input data
            
        Returns:
            np.ndarray: Predicted class labels
        """
        probabilities = self.predict(x)
        return np.argmax(probabilities, axis=1)
    
    def compute_loss(self, predictions, targets):
        """
        Compute loss for given predictions and targets.
        
        Args:
            predictions (np.ndarray): Model predictions
            targets (np.ndarray): True targets
            
        Returns:
            float: Loss value
        """
        return self.loss_function.forward(predictions, targets)
    
    def compute_accuracy(self, predictions, targets):
        """
        Compute accuracy for given predictions and targets.
        
        Args:
            predictions (np.ndarray): Model predictions (probabilities)
            targets (np.ndarray): True targets (one-hot or class indices)
            
        Returns:
            float: Accuracy score
        """
        # Convert one-hot to class indices if necessary
        if targets.ndim > 1 and targets.shape[1] > 1:
            true_classes = np.argmax(targets, axis=1)
        else:
            true_classes = targets
        
        predicted_classes = np.argmax(predictions, axis=1)
        return accuracy_score(true_classes, predicted_classes)
    
    def train_step(self, x_batch, y_batch):
        """
        Perform one training step on a batch of data.
        
        Args:
            x_batch (np.ndarray): Input batch
            y_batch (np.ndarray): Target batch
            
        Returns:
            tuple: (loss, accuracy)
        """
        # Forward pass
        predictions = self.forward(x_batch)
        
        # Compute loss
        loss = self.compute_loss(predictions, y_batch)
        
        # Compute accuracy
        accuracy = self.compute_accuracy(predictions, y_batch)
        
        # Backward pass
        grad_loss = self.loss_function.backward(predictions, y_batch)
        self.backward(grad_loss)
        
        # Update weights
        self.update_weights()
        
        return loss, accuracy
    
    def evaluate(self, x, y):
        """
        Evaluate the model on given data.
        
        Args:
            x (np.ndarray): Input data
            y (np.ndarray): Target data
            
        Returns:
            tuple: (loss, accuracy)
        """
        predictions = self.predict(x)
        loss = self.compute_loss(predictions, y)
        accuracy = self.compute_accuracy(predictions, y)
        return loss, accuracy
    
    def get_confusion_matrix(self, x, y):
        """
        Compute confusion matrix for given data.
        
        Args:
            x (np.ndarray): Input data
            y (np.ndarray): True targets
            
        Returns:
            np.ndarray: Confusion matrix
        """
        predictions = self.predict_classes(x)
        
        # Convert one-hot to class indices if necessary
        if y.ndim > 1 and y.shape[1] > 1:
            true_classes = np.argmax(y, axis=1)
        else:
            true_classes = y
        
        # Determine number of classes if output_size is None
        if self.output_size is None:
            num_classes = max(np.max(true_classes), np.max(predictions)) + 1
        else:
            num_classes = self.output_size
        
        return confusion_matrix(true_classes, predictions, num_classes)
    
    def save_model(self, filepath):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'architecture': {
                'input_size': self.input_size,
                'hidden_layers': self.hidden_layers,
                'output_size': self.output_size,
                'use_dropout': self.use_dropout,
                'dropout_rate': self.dropout_rate
            },
            'parameters': [],
            'history': self.history,
            'learning_rate': self.learning_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save layer parameters
        for layer in self.layers:
            if hasattr(layer, 'get_params'):
                model_data['parameters'].append(layer.get_params())
            else:
                model_data['parameters'].append(None)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    def load_model(self, filepath):
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore architecture
        arch = model_data['architecture']
        self.input_size = arch['input_size']
        self.hidden_layers = arch['hidden_layers']
        self.output_size = arch['output_size']
        self.use_dropout = arch.get('use_dropout', False)
        self.dropout_rate = arch.get('dropout_rate', 0.5)
        
        # Restore other attributes
        self.learning_rate = model_data['learning_rate']
        self.history = model_data['history']
        
        # Clear existing layers
        self.layers = []
        # Check if this model was built using constructor (has architecture info)
        # or using add_layer approach (architecture values are None)
        if (self.input_size is not None and self.hidden_layers is not None 
            and self.output_size is not None):
            # Model was built using constructor - rebuild using _build_network
            self._build_network('relu', 'softmax', 'xavier')
        else:
            # Model was built using add_layer approach - need to reconstruct layers
            # We'll create the layers from the saved parameters
            parameters = model_data['parameters']
            
            # Reconstruct layers based on saved parameters
            for i, param in enumerate(parameters):
                if param is None:
                    # This is a dropout layer
                    dropout_layer = DropoutLayer(dropout_rate=self.dropout_rate)
                    self.layers.append(dropout_layer)
                else:
                    # This is a dense layer - reconstruct from parameters
                    weights = param['weights']
                    bias = param.get('bias')  # Use get() since bias might not exist
                    input_size, output_size = weights.shape
                    
                    # Determine activation based on layer position and check if it's the last dense layer
                    # Count how many dense layers we have total
                    total_dense_layers = sum(1 for p in parameters if p is not None)
                    current_dense_layer = sum(1 for j in range(i+1) for p in [parameters[j]] if p is not None)
                    
                    if current_dense_layer == total_dense_layers:
                        # Last dense layer - use softmax
                        from .activations import Softmax
                        activation = Softmax()
                    else:
                        # Hidden layer - use ReLU
                        from .activations import ReLU
                        activation = ReLU()
                    
                    # Create dense layer
                    layer = DenseLayer(input_size, output_size, activation=activation)
                    self.layers.append(layer)
        
        # Restore layer parameters
        parameters = model_data['parameters']
        param_idx = 0
        
        for layer in self.layers:
            if hasattr(layer, 'set_params'):
                # Find the next non-None parameter
                while param_idx < len(parameters) and parameters[param_idx] is None:
                    param_idx += 1
                
                if param_idx < len(parameters):
                    layer.set_params(parameters[param_idx])
                    param_idx += 1
        
        print(f"Model loaded from {filepath}")
    
    def summary(self):
        """Print a summary of the network architecture."""
        print("Neural Network Summary")
        print("=" * 50)
        print(f"Input size: {self.input_size}")
        
        total_params = 0
        layer_num = 1
        
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                params = layer.weights.size
                if hasattr(layer, 'bias') and layer.bias is not None:
                    params += layer.bias.size
                
                print(f"Layer {layer_num}: {layer}")
                print(f"  Parameters: {params:,}")
                total_params += params
                layer_num += 1
            elif hasattr(layer, 'dropout_rate'):
                print(f"Layer {layer_num}: {layer}")
                layer_num += 1
        
        print("=" * 50)
        print(f"Total parameters: {total_params:,}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Loss function: {self.loss_function.__class__.__name__}")
    
    def __repr__(self):
        """String representation of the neural network."""
        return (f"NeuralNetwork(input_size={self.input_size}, "
                f"hidden_layers={self.hidden_layers}, "
                f"output_size={self.output_size})")
