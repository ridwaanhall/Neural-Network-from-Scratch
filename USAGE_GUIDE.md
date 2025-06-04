# Neural Network from Scratch - Usage Guide

## 🎯 Complete MNIST Neural Network Implementation

This project provides a professional implementation of a neural network from scratch using only NumPy for MNIST digit classification.

## 🚀 Quick Start

### 1. Run the Complete Pipeline

```bash
# Full training with default settings
python main.py

# Quick test (reduced dataset and epochs)
python main.py --quick_test

# Use different architectures
python main.py --architecture deep --epochs 100

# Skip plot generation
python main.py --no_plots
```

### 2. Custom Training

```bash
# Custom training script
python train.py

# Custom evaluation
python test.py --model_path models/your_model.pkl
```

### 3. Minimal Testing

```bash
# Basic functionality test
python minimal_test.py

# Component testing
python test_basic.py
```

## 📁 Project Structure

```txt
nn-scratch/
├── src/
│   ├── data/
│   │   └── data_loader.py          # MNIST data loading and preprocessing
│   ├── models/
│   │   ├── activations.py          # Activation functions (ReLU, Sigmoid, etc.)
│   │   ├── layers.py               # Neural network layers (Dense, Dropout)
│   │   └── neural_network.py       # Main NeuralNetwork class
│   ├── training/
│   │   ├── loss_functions.py       # Loss functions (Cross-entropy, MSE, etc.)
│   │   └── trainer.py              # Training logic with advanced features
│   └── utils/
│       ├── metrics.py              # Evaluation metrics
│       └── visualization.py        # Plotting utilities
├── data/                           # MNIST data storage (auto-created)
├── models/                         # Saved models (auto-created)
├── logs/                           # Training logs and plots (auto-created)
├── main.py                         # Complete end-to-end pipeline
├── train.py                        # Standalone training script
├── test.py                         # Standalone evaluation script
├── minimal_test.py                 # Quick functionality test
└── requirements.txt                # Dependencies
```

## 🧠 Model Architectures

### Simple (2 layers)

- Input (784) → Dense(128, ReLU) → Dense(10, Softmax)
- Fast training, good for testing

### Default (3 layers with dropout)

- Input (784) → Dense(256, ReLU) → Dropout(0.2) → Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)
- Balanced performance and training time

### Deep (4 layers with dropout)

- Input (784) → Dense(512, ReLU) → Dropout(0.3) → Dense(256, ReLU) → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)
- Higher capacity, longer training time

## ⚙️ Features

### Core Neural Network

- ✅ **From-scratch implementation** using only NumPy
- ✅ **Multiple activation functions**: ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
- ✅ **Multiple layer types**: Dense, Dropout
- ✅ **Weight initialization**: Xavier, He, Random
- ✅ **Backpropagation** with automatic gradient computation

### Training Features

- ✅ **Advanced optimizer**: SGD with momentum
- ✅ **Learning rate scheduling**: Step decay, exponential decay, plateau
- ✅ **Early stopping** with patience
- ✅ **Batch training** with configurable batch sizes
- ✅ **Multiple loss functions**: Cross-entropy, MSE, Binary cross-entropy, Huber

### Data & Evaluation

- ✅ **Automatic MNIST download** and preprocessing
- ✅ **Comprehensive metrics**: Accuracy, Precision, Recall, F1-Score
- ✅ **Confusion matrix** generation
- ✅ **Training visualization**: Loss/accuracy curves
- ✅ **Sample prediction visualization**

### Production Features

- ✅ **Model saving/loading** with pickle
- ✅ **Comprehensive logging** with timestamps
- ✅ **Progress tracking** during training
- ✅ **Error handling** and validation
- ✅ **Command-line interface** with arguments

## 📊 Expected Performance

- **Simple model**: ~85-90% accuracy in 10-20 epochs
- **Default model**: ~92-95% accuracy in 30-50 epochs  
- **Deep model**: ~95-97% accuracy in 50-100 epochs

## 🔧 Configuration Options

### Command Line Arguments

```bash
python main.py [OPTIONS]

Options:
  --architecture {simple,default,deep}   Model architecture (default: default)
  --epochs INT                           Number of training epochs (default: 50)
  --batch_size INT                       Batch size (default: 128)
  --learning_rate FLOAT                  Learning rate (default: 0.001)
  --no_plots                            Skip plot generation
  --quick_test                          Run with reduced dataset for testing
```

### Programming Interface

```python
from src.models.neural_network import NeuralNetwork
from src.models.layers import DenseLayer, DropoutLayer
from src.models.activations import ReLU, Softmax

# Create custom model
model = NeuralNetwork()
model.add_layer(DenseLayer(784, 256, activation=ReLU(), weight_init='xavier'))
model.add_layer(DropoutLayer(0.2))
model.add_layer(DenseLayer(256, 10, activation=Softmax(), weight_init='xavier'))

# Train with custom settings
from src.training.trainer import Trainer
from src.training.loss_functions import CrossEntropyLoss

trainer = Trainer(
    model=model,
    loss_function=CrossEntropyLoss(),
    learning_rate=0.001,
    batch_size=128,
    momentum=0.9
)

history = trainer.train(X_train, y_train, X_val, y_val, epochs=50)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're running from the project root directory
2. **Memory issues**: Use smaller batch sizes or the `--quick_test` flag
3. **Slow training**: Use the simple architecture or fewer epochs for testing
4. **Missing plots**: Install matplotlib with `pip install matplotlib`

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run minimal test first
python minimal_test.py
```

## 📈 Extending the Implementation

### Add New Activation Functions

```python
# In src/models/activations.py
class YourActivation(ActivationFunction):
    def forward(self, x):
        # Your forward implementation
        pass
    
    def backward(self, x):
        # Your backward implementation  
        pass
```

### Add New Layer Types

```python
# In src/models/layers.py
class YourLayer(Layer):
    def forward(self, x, training=True):
        # Your forward implementation
        pass
    
    def backward(self, grad_output):
        # Your backward implementation
        pass
```

## 📝 Notes

- **Pure NumPy**: No TensorFlow, PyTorch, or scikit-learn used for core ML operations
- **Educational**: Designed for learning neural network fundamentals
- **Production-ready**: Includes proper error handling, logging, and validation
- **Extensible**: Clean OOP design for easy modification and extension

## 🎓 Learning Outcomes

By studying this implementation, you'll understand:

- How neural networks work mathematically
- Backpropagation algorithm implementation  
- Gradient descent optimization
- Weight initialization strategies
- Regularization techniques (dropout)
- Training best practices
- Model evaluation and metrics
- Software engineering for ML projects
