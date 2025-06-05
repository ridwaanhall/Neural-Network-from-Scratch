"""
Visualization Utilities

This module contains various visualization functions for neural network training
and evaluation, including training curves, confusion matrices, and sample predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import os


def plot_training_history(history, save_path=None, figsize=(12, 4)):
    """
    Plot training and validation loss and accuracy curves.
    
    Args:
        history (dict): Training history containing losses and accuracies
        save_path (str): Path to save the plot (optional)
        figsize (tuple): Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if history['val_loss'] and any(loss > 0 for loss in history['val_loss']):
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    if history['val_accuracy'] and any(acc > 0 for acc in history['val_accuracy']):
        ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    else:
        plt.show()


def plot_confusion_matrix(confusion_matrix, class_names=None, save_path=None, 
                         figsize=(8, 6), normalize=False, cmap='Blues'):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix
        class_names (list): Names of classes (optional)
        save_path (str): Path to save the plot (optional)
        figsize (tuple): Figure size
        normalize (bool): Whether to normalize the confusion matrix
        cmap (str): Colormap for the heatmap
    """
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm = confusion_matrix
        fmt = 'd'
        title = 'Confusion Matrix'
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(confusion_matrix))]
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    
    plt.show()


def plot_sample_predictions(X_test, y_true, y_pred, num_samples=16, 
                          save_path=None, figsize=(12, 8), class_names=None):
    """
    Plot sample predictions with true and predicted labels.
    
    Args:
        X_test (np.ndarray): Test images (flattened or 2D)
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        num_samples (int): Number of samples to display
        save_path (str): Path to save the plot (optional)
        figsize (tuple): Figure size
        class_names (list): Names of classes (optional)
    """
    # Reshape images if flattened
    if X_test.ndim == 2 and X_test.shape[1] == 784:  # MNIST case
        images = X_test.reshape(-1, 28, 28)
    else:
        images = X_test
    
    # Select random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    # Calculate grid size
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Display image
        if images.ndim == 3:  # Grayscale
            ax.imshow(images[idx], cmap='gray')
        else:  # Color
            ax.imshow(images[idx])
        
        # Set title with true and predicted labels
        true_label = y_true[idx] if y_true.ndim == 1 else np.argmax(y_true[idx])
        pred_label = y_pred[idx] if y_pred.ndim == 1 else np.argmax(y_pred[idx])
        
        if class_names:
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
        else:
            true_name = str(true_label)
            pred_name = str(pred_label)
        
        # Color title based on correctness
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_name}\nPred: {pred_name}', 
                    fontsize=10, color=color, fontweight='bold')
        
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    
    plt.show()


def plot_class_distribution(y_data, class_names=None, save_path=None, 
                          figsize=(10, 6), title="Class Distribution"):
    """
    Plot class distribution as a bar chart.
    
    Args:
        y_data (np.ndarray): Labels (one-hot or class indices)
        class_names (list): Names of classes (optional)
        save_path (str): Path to save the plot (optional)
        figsize (tuple): Figure size
        title (str): Plot title
    """
    # Convert one-hot to class indices if necessary
    if y_data.ndim > 1 and y_data.shape[1] > 1:
        labels = np.argmax(y_data, axis=1)
    else:
        labels = y_data
    
    # Count classes
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique_labels]
    else:
        class_names = [class_names[i] for i in unique_labels]
      # Create bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert class names to strings and ensure they're treated as categorical
    x_positions = range(len(class_names))
    bars = ax.bar(x_positions, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_names)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Samples')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels if needed
    if len(class_names[0]) > 5:
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
    
    plt.show()


def plot_learning_curves(train_sizes, train_scores, val_scores, 
                        save_path=None, figsize=(10, 6)):
    """
    Plot learning curves showing performance vs training set size.
    
    Args:
        train_sizes (np.ndarray): Training set sizes
        train_scores (np.ndarray): Training scores for each size
        val_scores (np.ndarray): Validation scores for each size
        save_path (str): Path to save the plot (optional)
        figsize (tuple): Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                    alpha=0.1, color='blue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                    alpha=0.1, color='red')
    
    ax.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curves plot saved to {save_path}")
    
    plt.show()


def plot_activation_distributions(activations, layer_names=None, 
                                save_path=None, figsize=(15, 10)):
    """
    Plot distributions of activations for each layer.
    
    Args:
        activations (list): List of activation arrays for each layer
        layer_names (list): Names of layers (optional)
        save_path (str): Path to save the plot (optional)
        figsize (tuple): Figure size
    """
    num_layers = len(activations)
    rows = int(np.ceil(num_layers / 3))
    cols = min(3, num_layers)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_layers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    for i, activation in enumerate(activations):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Flatten activation if needed
        flat_activation = activation.flatten()
        
        # Plot histogram
        ax.hist(flat_activation, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Set title
        title = layer_names[i] if layer_names else f'Layer {i+1}'
        ax.set_title(f'{title} Activations', fontweight='bold')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(flat_activation)
        std_val = np.std(flat_activation)
        ax.axvline(mean_val, color='red', linestyle='--', 
                  label=f'Mean: {mean_val:.3f}')
        ax.text(0.7, 0.9, f'Std: {std_val:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend()
    
    # Hide unused subplots
    for i in range(num_layers, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Activation distributions plot saved to {save_path}")
    
    plt.show()


def plot_weights_distribution(model, save_path=None, figsize=(15, 10)):
    """
    Plot weight distributions for each layer in the model.
    
    Args:
        model: Neural network model
        save_path (str): Path to save the plot (optional)
        figsize (tuple): Figure size
    """
    # Extract weights from each layer
    weights = []
    layer_names = []
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'weights'):
            weights.append(layer.weights)
            layer_names.append(f'Layer {i+1}')
    
    if not weights:
        print("No weights found in the model.")
        return
    
    num_layers = len(weights)
    rows = int(np.ceil(num_layers / 3))
    cols = min(3, num_layers)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_layers == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    for i, weight_matrix in enumerate(weights):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Flatten weights
        flat_weights = weight_matrix.flatten()
        
        # Plot histogram
        ax.hist(flat_weights, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        
        # Set title
        ax.set_title(f'{layer_names[i]} Weights', fontweight='bold')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(flat_weights)
        std_val = np.std(flat_weights)
        ax.axvline(mean_val, color='blue', linestyle='--', 
                  label=f'Mean: {mean_val:.4f}')
        ax.text(0.7, 0.9, f'Std: {std_val:.4f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.legend()
    
    # Hide unused subplots
    for i in range(num_layers, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weight distributions plot saved to {save_path}")
    
    plt.show()


def create_visualization_report(model, history, X_test, y_test, y_pred, 
                              confusion_mat, save_dir='logs', timestamp=None, run_type='train'):
    """
    Create a comprehensive visualization report for model evaluation.
    
    Args:
        model: Trained neural network model
        history (dict): Training history
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): True test labels
        y_pred (numpy.ndarray): Predicted test labels
        confusion_mat (numpy.ndarray): Confusion matrix
        save_dir (str): Directory to save visualizations
        timestamp (str): Optional timestamp for file naming
        run_type (str): Type of run ('train' or 'test') for directory naming
    """
    # Generate timestamp if not provided
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create timestamped subdirectory for this run with run type
    run_dir = os.path.join(save_dir, f'run_{run_type}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print("Creating visualization report...")
    print(f"Saving visualizations to: {run_dir}")
    
    # 1. Training history (only for training runs)
    if run_type == 'train' and history is not None:
        plot_training_history(history, 
                             save_path=os.path.join(run_dir, 'training_history.png'))
        print(f"Training history plot saved to {run_dir}/training_history.png")
    
    # 2. Confusion matrix
    class_names = [str(i) for i in range(10)]  # MNIST digits
    plot_confusion_matrix(confusion_mat, class_names=class_names,
                         save_path=os.path.join(run_dir, 'confusion_matrix.png'))
    print(f"Confusion matrix plot saved to {run_dir}/confusion_matrix.png")
    
    # 3. Normalized confusion matrix
    plot_confusion_matrix(confusion_mat, class_names=class_names, normalize=True,
                         save_path=os.path.join(run_dir, 'confusion_matrix_normalized.png'))
    print(f"Normalized confusion matrix plot saved to {run_dir}/confusion_matrix_normalized.png")
    
    # 4. Sample predictions
    plot_sample_predictions(X_test, y_test, y_pred, num_samples=16,
                          save_path=os.path.join(run_dir, 'sample_predictions.png'),
                          class_names=class_names)
    print(f"Sample predictions plot saved to {run_dir}/sample_predictions.png")
    
    # 5. Class distribution
    title = f"{'Test' if run_type == 'test' else 'Test'} Set Class Distribution"
    plot_class_distribution(y_test, class_names=class_names,
                          save_path=os.path.join(run_dir, 'class_distribution.png'),
                          title=title)
    print(f"Class distribution plot saved to {run_dir}/class_distribution.png")
    
    # 6. Weight distributions (only for training runs)
    if run_type == 'train':
        plot_weights_distribution(model, 
                                save_path=os.path.join(run_dir, 'weight_distributions.png'))
        print(f"Weight distributions plot saved to {run_dir}/weight_distributions.png")
      # Create a summary file with run information
    summary_path = os.path.join(run_dir, f'{run_type}_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"{run_type.capitalize()} Run Summary - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{run_type.capitalize()} completed at: {timestamp}\n")
        if run_type == 'train':
            f.write(f"Total epochs: {len(history['train_loss'])}\n")
            if history['train_loss']:
                f.write(f"Final training loss: {history['train_loss'][-1]:.4f}\n")
                f.write(f"Final training accuracy: {history['train_accuracy'][-1]:.4f}\n")
            if history['val_loss'] and any(loss > 0 for loss in history['val_loss']):
                f.write(f"Final validation loss: {history['val_loss'][-1]:.4f}\n")
                f.write(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}\n")
        f.write(f"\nVisualization files:\n")
        if run_type == 'train':
            f.write(f"- training_history.png\n")
        f.write(f"- confusion_matrix.png\n")
        f.write(f"- confusion_matrix_normalized.png\n")
        f.write(f"- sample_predictions.png\n")
        f.write(f"- class_distribution.png\n")
        if run_type == 'train':
            f.write(f"- weight_distributions.png\n")
    
    print(f"{run_type.capitalize()} summary saved to {run_dir}/{run_type}_summary.txt")
    print(f"Visualization report completed! All files saved in: {run_dir}")
    
    return run_dir


# Set default matplotlib style for better plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
