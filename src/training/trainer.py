"""
Neural Network Trainer Module

This module contains the trainer class that handles the complete training process
including epoch management, validation, early stopping, and learning rate scheduling.
"""

import numpy as np
import time
from datetime import datetime
from ..data.data_loader import DataPreprocessor
from ..utils.metrics import accuracy_score
from ..utils.visualization import plot_training_history


class Trainer:
    """
    Neural network trainer with advanced training features.
    
    Handles the complete training process including:
    - Mini-batch training with momentum
    - Validation monitoring
    - Early stopping
    - Learning rate scheduling
    - Training progress tracking
    """
    
    def __init__(self, model, patience=10, min_delta=1e-4, 
                 lr_scheduler=None, save_best=True, verbose=1):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model to train
            patience (int): Number of epochs with no improvement before early stopping
            min_delta (float): Minimum change to qualify as an improvement
            lr_scheduler (dict): Learning rate scheduler configuration
            save_best (bool): Whether to save the best model during training
            verbose (int): Verbosity level (0: silent, 1: progress bar, 2: one line per epoch)
        """
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.lr_scheduler = lr_scheduler
        self.save_best = save_best
        self.verbose = verbose
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_model_params = None
        self.epochs_without_improvement = 0
        self.current_lr = model.learning_rate
        
        # Training metrics
        self.start_time = None
        self.epoch_times = []
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=32, momentum=0.9, 
              validation_split=0.2, shuffle=True):
        """
        Train the neural network.
        
        Args:
            X_train (np.ndarray): Training input data
            y_train (np.ndarray): Training target data
            X_val (np.ndarray): Validation input data (optional)
            y_val (np.ndarray): Validation target data (optional)
            epochs (int): Maximum number of training epochs
            batch_size (int): Size of mini-batches
            momentum (float): Momentum coefficient for optimization
            validation_split (float): Fraction of training data to use for validation
            shuffle (bool): Whether to shuffle training data each epoch
            
        Returns:
            dict: Training history containing losses and accuracies
        """
        # Prepare validation data
        if X_val is None or y_val is None:
            if validation_split > 0:
                X_train, X_val, y_train, y_val = DataPreprocessor.train_validation_split(
                    X_train, y_train, validation_split=validation_split, shuffle=shuffle
                )
            else:
                X_val, y_val = None, None
        
        # Initialize training
        self.start_time = time.time()
        self.model.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        if self.verbose > 0:
            print(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Training samples: {len(X_train)}")
            if X_val is not None:
                print(f"Validation samples: {len(X_val)}")
            print(f"Batch size: {batch_size}")
            print(f"Initial learning rate: {self.current_lr}")
            print("-" * 60)
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(
                X_train, y_train, batch_size, momentum, shuffle
            )
            
            # Validation phase
            if X_val is not None:
                val_loss, val_acc = self.model.evaluate(X_val, y_val)
            else:
                val_loss, val_acc = 0.0, 0.0
            
            # Record metrics
            self.model.history['train_loss'].append(train_loss)
            self.model.history['train_accuracy'].append(train_acc)
            self.model.history['val_loss'].append(val_loss)
            self.model.history['val_accuracy'].append(val_acc)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # Update learning rate
            self._update_learning_rate(epoch, val_loss)
            
            # Print progress
            if self.verbose > 0:
                self._print_epoch_results(epoch, epochs, train_loss, train_acc, 
                                        val_loss, val_acc, epoch_time)
            
            # Early stopping and model saving
            if X_val is not None:
                if self._check_early_stopping(val_loss):
                    if self.verbose > 0:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
                
                if self.save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model_params = self._get_model_params()
        
        # Restore best model if save_best is enabled
        if self.save_best and self.best_model_params is not None:
            self._set_model_params(self.best_model_params)
            if self.verbose > 0:
                print(f"\nRestored best model with validation loss: {self.best_val_loss:.4f}")
        
        # Training summary
        total_time = time.time() - self.start_time
        if self.verbose > 0:
            self._print_training_summary(total_time)
        
        return self.model.history
    
    def _train_epoch(self, X_train, y_train, batch_size, momentum, shuffle):
        """
        Train for one epoch.
        
        Args:
            X_train (np.ndarray): Training input data
            y_train (np.ndarray): Training target data
            batch_size (int): Size of mini-batches
            momentum (float): Momentum coefficient
            shuffle (bool): Whether to shuffle data
            
        Returns:
            tuple: (average_loss, average_accuracy)
        """
        epoch_losses = []
        epoch_accuracies = []
        
        # Create mini-batches
        batches = DataPreprocessor.create_mini_batches(
            X_train, y_train, batch_size, shuffle
        )
        
        # Train on each batch
        for X_batch, y_batch in batches:
            loss, accuracy = self.model.train_step(X_batch, y_batch)
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
        
        return np.mean(epoch_losses), np.mean(epoch_accuracies)
    
    def _update_learning_rate(self, epoch, val_loss):
        """
        Update learning rate based on scheduler configuration.
        
        Args:
            epoch (int): Current epoch number
            val_loss (float): Current validation loss
        """
        if self.lr_scheduler is None:
            return
        
        scheduler_type = self.lr_scheduler.get('type', 'step')
        
        if scheduler_type == 'step':
            # Step decay
            step_size = self.lr_scheduler.get('step_size', 10)
            gamma = self.lr_scheduler.get('gamma', 0.1)
            
            if (epoch + 1) % step_size == 0:
                self.current_lr *= gamma
                self.model.learning_rate = self.current_lr
        
        elif scheduler_type == 'exponential':
            # Exponential decay
            gamma = self.lr_scheduler.get('gamma', 0.95)
            self.current_lr *= gamma
            self.model.learning_rate = self.current_lr
        
        elif scheduler_type == 'plateau':
            # Reduce on plateau
            factor = self.lr_scheduler.get('factor', 0.5)
            patience = self.lr_scheduler.get('patience', 5)
            
            if self.epochs_without_improvement >= patience:
                self.current_lr *= factor
                self.model.learning_rate = self.current_lr
                self.epochs_without_improvement = 0  # Reset counter
    
    def _check_early_stopping(self, val_loss):
        """
        Check if early stopping criteria are met.
        
        Args:
            val_loss (float): Current validation loss
            
        Returns:
            bool: True if early stopping should be triggered
        """
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            return self.epochs_without_improvement >= self.patience
    
    def _get_model_params(self):
        """Get current model parameters."""
        params = []
        for layer in self.model.layers:
            if hasattr(layer, 'get_params'):
                params.append(layer.get_params())
            else:
                params.append(None)
        return params
    
    def _set_model_params(self, params):
        """Set model parameters."""
        for i, layer in enumerate(self.model.layers):
            if hasattr(layer, 'set_params') and params[i] is not None:
                layer.set_params(params[i])
    
    def _print_epoch_results(self, epoch, total_epochs, train_loss, train_acc, 
                           val_loss, val_acc, epoch_time):
        """Print results for current epoch."""
        if self.verbose == 1:
            # Progress bar style
            progress = (epoch + 1) / total_epochs
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            print(f"\rEpoch {epoch + 1:3d}/{total_epochs} [{bar}] "
                  f"- {epoch_time:.2f}s - "
                  f"loss: {train_loss:.4f} - acc: {train_acc:.4f}", end='')
            
            if val_loss > 0:
                print(f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}", end='')
            
            if epoch == total_epochs - 1:
                print()  # New line at the end
        
        elif self.verbose == 2:
            # One line per epoch
            print(f"Epoch {epoch + 1:3d}/{total_epochs} - "
                  f"{epoch_time:.2f}s - "
                  f"loss: {train_loss:.4f} - acc: {train_acc:.4f}", end='')
            
            if val_loss > 0:
                print(f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            else:
                print()
    
    def _print_training_summary(self, total_time):
        """Print training summary."""
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Average time per epoch: {np.mean(self.epoch_times):.2f} seconds")
        
        history = self.model.history
        if history['train_loss']:
            final_train_loss = history['train_loss'][-1]
            final_train_acc = history['train_accuracy'][-1]
            print(f"Final training loss: {final_train_loss:.4f}")
            print(f"Final training accuracy: {final_train_acc:.4f}")
        
        if history['val_loss'] and any(loss > 0 for loss in history['val_loss']):
            final_val_loss = history['val_loss'][-1]
            final_val_acc = history['val_accuracy'][-1]
            print(f"Final validation loss: {final_val_loss:.4f}")
            print(f"Final validation accuracy: {final_val_acc:.4f}")
    
    def plot_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path (str): Path to save the plot (optional)
        """
        plot_training_history(self.model.history, save_path)
    
    def evaluate_model(self, X_test, y_test, verbose=True):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (np.ndarray): Test input data
            y_test (np.ndarray): Test target data
            verbose (bool): Whether to print results
            
        Returns:
            dict: Evaluation metrics
        """
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        
        # Get confusion matrix
        confusion_mat = self.model.get_confusion_matrix(X_test, y_test)
        
        # Calculate per-class accuracy
        per_class_acc = np.diag(confusion_mat) / np.sum(confusion_mat, axis=1)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'confusion_matrix': confusion_mat,
            'per_class_accuracy': per_class_acc
        }
        
        if verbose:
            print("\nTest Results:")
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print("\nPer-class Accuracy:")
            for i, acc in enumerate(per_class_acc):
                print(f"  Class {i}: {acc:.4f}")
        
        return results
