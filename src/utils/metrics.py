"""
Metrics and Evaluation Utilities

This module contains various metrics and evaluation functions for neural networks.
Includes accuracy, precision, recall, F1-score, and confusion matrix calculations.
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy score.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        
    Returns:
        float: Accuracy score
    """
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, num_classes, average='macro'):
    """
    Calculate precision score.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        num_classes (int): Number of classes
        average (str): Averaging method ('macro', 'micro', 'weighted', or None)
        
    Returns:
        float or np.ndarray: Precision score(s)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, num_classes)
    
    # Calculate precision for each class
    precisions = np.zeros(num_classes)
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        precisions[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    if average == 'macro':
        return np.mean(precisions)
    elif average == 'micro':
        tp_total = np.sum(np.diag(cm))
        fp_total = np.sum(cm) - tp_total
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    elif average == 'weighted':
        weights = np.sum(cm, axis=1)
        return np.average(precisions, weights=weights)
    else:
        return precisions


def recall_score(y_true, y_pred, num_classes, average='macro'):
    """
    Calculate recall score.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        num_classes (int): Number of classes
        average (str): Averaging method ('macro', 'micro', 'weighted', or None)
        
    Returns:
        float or np.ndarray: Recall score(s)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, num_classes)
    
    # Calculate recall for each class
    recalls = np.zeros(num_classes)
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        recalls[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if average == 'macro':
        return np.mean(recalls)
    elif average == 'micro':
        tp_total = np.sum(np.diag(cm))
        fn_total = np.sum(cm) - tp_total
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    elif average == 'weighted':
        weights = np.sum(cm, axis=1)
        return np.average(recalls, weights=weights)
    else:
        return recalls


def f1_score(y_true, y_pred, num_classes, average='macro'):
    """
    Calculate F1 score.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        num_classes (int): Number of classes
        average (str): Averaging method ('macro', 'micro', 'weighted', or None)
        
    Returns:
        float or np.ndarray: F1 score(s)
    """
    precision = precision_score(y_true, y_pred, num_classes, average=None)
    recall = recall_score(y_true, y_pred, num_classes, average=None)
    
    # Calculate F1 for each class
    f1_scores = np.zeros(num_classes)
    for i in range(num_classes):
        if precision[i] + recall[i] > 0:
            f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1_scores[i] = 0
    
    if average == 'macro':
        return np.mean(f1_scores)
    elif average == 'micro':
        prec_micro = precision_score(y_true, y_pred, num_classes, average='micro')
        rec_micro = recall_score(y_true, y_pred, num_classes, average='micro')
        if prec_micro + rec_micro > 0:
            return 2 * (prec_micro * rec_micro) / (prec_micro + rec_micro)
        else:
            return 0
    elif average == 'weighted':
        weights = np.bincount(y_true, minlength=num_classes)
        return np.average(f1_scores, weights=weights)
    else:
        return f1_scores


def confusion_matrix(y_true, y_pred, num_classes):
    """
    Calculate confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        num_classes (int): Number of classes
        
    Returns:
        np.ndarray: Confusion matrix of shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            cm[true_label, pred_label] += 1
    return cm


def classification_report(y_true, y_pred, num_classes, class_names=None):
    """
    Generate a comprehensive classification report.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        num_classes (int): Number of classes
        class_names (list): Names of classes (optional)
        
    Returns:
        str: Formatted classification report
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, num_classes, average=None)
    recall = recall_score(y_true, y_pred, num_classes, average=None)
    f1 = f1_score(y_true, y_pred, num_classes, average=None)
    support = np.bincount(y_true, minlength=num_classes)
    
    # Calculate averages
    macro_avg_prec = np.mean(precision)
    macro_avg_rec = np.mean(recall)
    macro_avg_f1 = np.mean(f1)
    
    weighted_avg_prec = np.average(precision, weights=support)
    weighted_avg_rec = np.average(recall, weights=support)
    weighted_avg_f1 = np.average(f1, weights=support)
    
    accuracy = accuracy_score(y_true, y_pred)
    
    # Format report
    report = "Classification Report\n"
    report += "=" * 60 + "\n"
    report += f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n"
    report += "-" * 60 + "\n"
    
    for i in range(num_classes):
        report += f"{class_names[i]:<15} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1[i]:<10.3f} {support[i]:<10}\n"
    
    report += "-" * 60 + "\n"
    report += f"{'Accuracy':<15} {'':<10} {'':<10} {accuracy:<10.3f} {np.sum(support):<10}\n"
    report += f"{'Macro Avg':<15} {macro_avg_prec:<10.3f} {macro_avg_rec:<10.3f} {macro_avg_f1:<10.3f} {np.sum(support):<10}\n"
    report += f"{'Weighted Avg':<15} {weighted_avg_prec:<10.3f} {weighted_avg_rec:<10.3f} {weighted_avg_f1:<10.3f} {np.sum(support):<10}\n"
    
    return report


def top_k_accuracy(y_true, y_pred_proba, k=1):
    """
    Calculate top-k accuracy.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        k (int): Number of top predictions to consider
        
    Returns:
        float: Top-k accuracy
    """
    # Get top-k predictions
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    
    # Check if true label is in top-k predictions
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_preds[i]:
            correct += 1
    
    return correct / len(y_true)


def roc_curve_multiclass(y_true, y_pred_proba, num_classes):
    """
    Calculate ROC curve for multiclass classification.
    
    Args:
        y_true (np.ndarray): True labels (class indices)
        y_pred_proba (np.ndarray): Predicted probabilities
        num_classes (int): Number of classes
        
    Returns:
        dict: ROC curves for each class
    """
    # Convert to one-hot if necessary
    if y_true.ndim == 1:
        y_true_onehot = np.eye(num_classes)[y_true]
    else:
        y_true_onehot = y_true
    
    roc_curves = {}
    
    for i in range(num_classes):
        # For each class, treat it as binary classification
        y_true_binary = y_true_onehot[:, i]
        y_pred_binary = y_pred_proba[:, i]
        
        # Sort by predicted probability
        sorted_indices = np.argsort(y_pred_binary)[::-1]
        y_true_sorted = y_true_binary[sorted_indices]
        y_pred_sorted = y_pred_binary[sorted_indices]
        
        # Calculate TPR and FPR
        tpr = []
        fpr = []
        
        for threshold in np.unique(y_pred_sorted):
            predictions = (y_pred_sorted >= threshold).astype(int)
            
            tp = np.sum((predictions == 1) & (y_true_sorted == 1))
            tn = np.sum((predictions == 0) & (y_true_sorted == 0))
            fp = np.sum((predictions == 1) & (y_true_sorted == 0))
            fn = np.sum((predictions == 0) & (y_true_sorted == 1))
            
            tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        
        roc_curves[f'class_{i}'] = {
            'fpr': np.array(fpr),
            'tpr': np.array(tpr),
            'auc': np.trapz(tpr, fpr) if len(fpr) > 1 else 0
        }
    
    return roc_curves


def calculate_class_weights(y_true, num_classes, method='balanced'):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y_true (np.ndarray): True labels
        num_classes (int): Number of classes
        method (str): Method to calculate weights ('balanced' or 'inverse_freq')
        
    Returns:
        np.ndarray: Class weights
    """
    class_counts = np.bincount(y_true, minlength=num_classes)
    
    if method == 'balanced':
        # Balanced weights: n_samples / (n_classes * class_count)
        total_samples = len(y_true)
        weights = total_samples / (num_classes * class_counts)
    elif method == 'inverse_freq':
        # Inverse frequency weights
        weights = 1.0 / class_counts
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights
    weights = weights / np.sum(weights) * num_classes
    
    return weights


class MetricsTracker:
    """
    Utility class to track and compute various metrics during training.
    """
    
    def __init__(self, num_classes, class_names=None):
        """
        Initialize metrics tracker.
        
        Args:
            num_classes (int): Number of classes
            class_names (list): Names of classes (optional)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.y_true = []
        self.y_pred = []
        self.y_pred_proba = []
    
    def update(self, y_true, y_pred, y_pred_proba=None):
        """
        Update metrics with new predictions.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray): Predicted probabilities (optional)
        """
        self.y_true.extend(y_true.flatten())
        self.y_pred.extend(y_pred.flatten())
        
        if y_pred_proba is not None:
            self.y_pred_proba.extend(y_pred_proba)
    
    def compute_metrics(self):
        """
        Compute all metrics.
        
        Returns:
            dict: Dictionary containing all computed metrics
        """
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, self.num_classes, 'macro'),
            'recall_macro': recall_score(y_true, y_pred, self.num_classes, 'macro'),
            'f1_macro': f1_score(y_true, y_pred, self.num_classes, 'macro'),
            'precision_weighted': precision_score(y_true, y_pred, self.num_classes, 'weighted'),
            'recall_weighted': recall_score(y_true, y_pred, self.num_classes, 'weighted'),
            'f1_weighted': f1_score(y_true, y_pred, self.num_classes, 'weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred, self.num_classes)
        }
        
        if self.y_pred_proba:
            y_pred_proba = np.array(self.y_pred_proba)
            metrics['top_1_accuracy'] = top_k_accuracy(y_true, y_pred_proba, k=1)
            metrics['top_2_accuracy'] = top_k_accuracy(y_true, y_pred_proba, k=2)
            metrics['top_3_accuracy'] = top_k_accuracy(y_true, y_pred_proba, k=3)
        
        return metrics
    
    def get_classification_report(self):
        """
        Get detailed classification report.
        
        Returns:
            str: Formatted classification report
        """
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        
        return classification_report(y_true, y_pred, self.num_classes, self.class_names)
