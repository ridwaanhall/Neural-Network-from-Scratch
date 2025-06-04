"""
Data Loading and Preprocessing Module

This module handles downloading, loading, and preprocessing of the MNIST dataset.
Includes utilities for data normalization, one-hot encoding, and train/validation splits.
"""

import numpy as np
import gzip
import pickle
import os
import requests
from urllib.parse import urljoin


class MNISTDataLoader:
    """
    MNIST dataset loader with automatic downloading and preprocessing.
    
    Downloads the MNIST dataset from the original source if not already present,
    and provides utilities for loading and preprocessing the data.
    """
    
    # MNIST dataset URLs
    BASE_URL = "http://yann.lecun.com/exdb/mnist/"
    FILES = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    def __init__(self, data_dir="data"):
        """
        Initialize MNIST loader.
        
        Args:
            data_dir (str): Directory to store MNIST data files
        """
        self.data_dir = data_dir
        self.ensure_data_dir()
    
    def ensure_data_dir(self):
        """Create data directory if it doesn't exist."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def download_file(self, filename):
        """
        Download a single MNIST file.
        
        Args:
            filename (str): Name of the file to download
        """
        url = urljoin(self.BASE_URL, filename)
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"{filename} already exists, skipping download.")
            return
        
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded {filename} successfully.")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            raise
    
    def download_mnist(self):
        """Download all MNIST files if they don't exist."""
        print("Checking MNIST dataset...")
        for file_type, filename in self.FILES.items():
            self.download_file(filename)
        print("MNIST dataset ready.")
    
    def load_images(self, filename):
        """
        Load images from MNIST IDX file format.
        
        Args:
            filename (str): Name of the images file
            
        Returns:
            np.ndarray: Array of images with shape (num_images, height, width)
        """
        filepath = os.path.join(self.data_dir, filename)
        
        with gzip.open(filepath, 'rb') as f:
            # Read magic number and dimensions
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            height = int.from_bytes(f.read(4), 'big')
            width = int.from_bytes(f.read(4), 'big')
            
            # Read image data
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, height, width)
            
        return images
    
    def load_labels(self, filename):
        """
        Load labels from MNIST IDX file format.
        
        Args:
            filename (str): Name of the labels file
            
        Returns:
            np.ndarray: Array of labels
        """
        filepath = os.path.join(self.data_dir, filename)
        
        with gzip.open(filepath, 'rb') as f:
            # Read magic number and number of labels
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            
            # Read label data
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            
        return labels
    def load_data(self, normalize=True, flatten=True, one_hot=True, validation_split=None):
        """
        Load and preprocess MNIST dataset.
        
        Args:
            normalize (bool): Whether to normalize pixel values to [0, 1]
            flatten (bool): Whether to flatten images to 1D vectors
            one_hot (bool): Whether to one-hot encode labels
            validation_split (float, optional): Fraction of training data to use for validation
            
        Returns:
            tuple: If validation_split is None: (X_train, y_train, X_test, y_test)
                   If validation_split is provided: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        # Download data if necessary
        self.download_mnist()
        
        # Load training data
        print("Loading training data...")
        X_train = self.load_images(self.FILES['train_images'])
        y_train = self.load_labels(self.FILES['train_labels'])
        
        # Load test data
        print("Loading test data...")
        X_test = self.load_images(self.FILES['test_images'])
        y_test = self.load_labels(self.FILES['test_labels'])
        
        print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples")
        
        # Preprocess data
        if normalize:
            X_train = X_train.astype(np.float32) / 255.0
            X_test = X_test.astype(np.float32) / 255.0
        
        if flatten:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        if one_hot:
            y_train = self.to_one_hot(y_train, 10)
            y_test = self.to_one_hot(y_test, 10)
          # Split training data into train/validation if requested
        if validation_split is not None:
            X_train_split, X_val, y_train_split, y_val = DataPreprocessor.train_validation_split(
                X_train, y_train, validation_split=validation_split, shuffle=True, random_seed=42
            )
            return (X_train_split, y_train_split), (X_val, y_val), (X_test, y_test)
        else:
            return X_train, y_train, X_test, y_test
    
    @staticmethod
    def to_one_hot(labels, num_classes):
        """
        Convert labels to one-hot encoding.
        
        Args:
            labels (np.ndarray): Array of class labels
            num_classes (int): Number of classes
            
        Returns:
            np.ndarray: One-hot encoded labels
        """
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot
    
    @staticmethod
    def from_one_hot(one_hot_labels):
        """
        Convert one-hot encoded labels back to class indices.
        
        Args:
            one_hot_labels (np.ndarray): One-hot encoded labels
            
        Returns:
            np.ndarray: Class indices
        """
        return np.argmax(one_hot_labels, axis=1)


class DataPreprocessor:
    """
    Data preprocessing utilities for neural network training.
    
    Provides various preprocessing techniques including normalization,
    standardization, and data augmentation.
    """
    
    @staticmethod
    def normalize(data, method='minmax'):
        """
        Normalize data using specified method.
        
        Args:
            data (np.ndarray): Input data
            method (str): Normalization method ('minmax' or 'zscore')
            
        Returns:
            tuple: (normalized_data, normalization_params)
        """
        if method == 'minmax':
            # Min-max normalization to [0, 1]
            data_min = np.min(data, axis=0)
            data_max = np.max(data, axis=0)
            
            # Avoid division by zero
            data_range = data_max - data_min
            data_range[data_range == 0] = 1
            
            normalized = (data - data_min) / data_range
            params = {'min': data_min, 'max': data_max, 'range': data_range}
            
        elif method == 'zscore':
            # Z-score normalization (standardization)
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            
            # Avoid division by zero
            std[std == 0] = 1
            
            normalized = (data - mean) / std
            params = {'mean': mean, 'std': std}
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized, params
    
    @staticmethod
    def apply_normalization(data, params, method='minmax'):
        """
        Apply previously computed normalization parameters to new data.
        
        Args:
            data (np.ndarray): Input data
            params (dict): Normalization parameters
            method (str): Normalization method used
            
        Returns:
            np.ndarray: Normalized data
        """
        if method == 'minmax':
            return (data - params['min']) / params['range']
        elif method == 'zscore':
            return (data - params['mean']) / params['std']
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def train_validation_split(X, y, validation_split=0.2, shuffle=True, random_seed=42):
        """
        Split training data into training and validation sets.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target labels
            validation_split (float): Fraction of data to use for validation
            shuffle (bool): Whether to shuffle data before splitting
            random_seed (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        num_samples = len(X)
        indices = np.arange(num_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        # Calculate split point
        val_size = int(num_samples * validation_split)
        train_size = num_samples - val_size
        
        # Split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Split data
        X_train = X[train_indices]
        X_val = X[val_indices]
        y_train = y[train_indices]
        y_val = y[val_indices]
        
        return X_train, X_val, y_train, y_val
    
    @staticmethod
    def create_mini_batches(X, y, batch_size=32, shuffle=True):
        """
        Create mini-batches from training data.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target labels
            batch_size (int): Size of each mini-batch
            shuffle (bool): Whether to shuffle data before batching
            
        Yields:
            tuple: (X_batch, y_batch) for each mini-batch
        """
        num_samples = len(X)
        indices = np.arange(num_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield X[batch_indices], y[batch_indices]
    
    @staticmethod
    def add_noise(data, noise_type='gaussian', noise_level=0.1):
        """
        Add noise to data for data augmentation.
        
        Args:
            data (np.ndarray): Input data
            noise_type (str): Type of noise ('gaussian' or 'uniform')
            noise_level (float): Magnitude of noise
            
        Returns:
            np.ndarray: Data with added noise
        """
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, data.shape)
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, data.shape)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        return data + noise
    
    @staticmethod
    def save_preprocessed_data(X_train, y_train, X_test, y_test, filepath):
        """
        Save preprocessed data to file.
        
        Args:
            X_train, y_train, X_test, y_test: Preprocessed datasets
            filepath (str): Path to save the data
        """
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Preprocessed data saved to {filepath}")
    
    @staticmethod
    def load_preprocessed_data(filepath):
        """
        Load preprocessed data from file.
        
        Args:
            filepath (str): Path to the saved data
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']
