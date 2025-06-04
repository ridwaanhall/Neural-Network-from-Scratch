# Neural Network from Scratch

A professional implementation of a neural network from scratch using only NumPy for MNIST digit classification.

## Project Structure

```txt
nn-scratch/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── neural_network.py
│   │   ├── layers.py
│   │   └── activations.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── loss_functions.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       └── visualization.py
├── data/
├── models/
├── logs/
├── main.py
├── train.py
├── test.py
└── requirements.txt
```

## Features

- Object-oriented design with clean separation of concerns
- Custom neural network implementation using only NumPy
- MNIST dataset support with automatic downloading
- Multiple activation functions (ReLU, Sigmoid, Softmax)
- Cross-entropy loss with backpropagation
- Training with mini-batch gradient descent
- Performance metrics and visualization
- Model saving and loading capabilities

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python train.py`
3. Test the model: `python test.py`
4. Run complete pipeline: `python main.py`

## Architecture

The neural network uses a feedforward architecture with:

- Input layer: 784 neurons (28x28 flattened images)
- Hidden layers: Configurable (default: 2 layers with 128 and 64 neurons)
- Output layer: 10 neurons (digits 0-9)
- Activation functions: ReLU for hidden layers, Softmax for output
- Loss function: Cross-entropy
- Optimizer: Mini-batch gradient descent with momentum
