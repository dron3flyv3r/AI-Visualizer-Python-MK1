# AI Visualizer Python Interface

A Python interface for working with neural network models trained on [aiviz.kasperlarsen.tech](https://aiviz.kasperlarsen.tech).

## Overview

This project allows you to load, use, and interact with neural network models that you've trained using the AI Visualizer web tool. After creating and training a model on the website, you can download the model file and use this Python interface to make predictions with it.

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd AI-Visualizer-Python-MK1
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## User Guide

### Step 1: Train a model on aiviz.kasperlarsen.tech
1. Visit [aiviz.kasperlarsen.tech](https://aiviz.kasperlarsen.tech)
2. Create your neural network by configuring the layers and neurons
3. Train the model using the web interface
4. Once satisfied with the model's performance, download the model as a `.txt` file

### Step 2: Load the model in Python
```python
from neural_network import NeuralNetwork

# Create a new neural network instance
model = NeuralNetwork()

# Load your downloaded model file
model.load_model("path/to/your/model.txt")

# Print information about the loaded model
print(model)
```

### Step 3: Make predictions using your model
```python
# Make a prediction with input values
# The number of inputs must match the input layer size of your model
result = model.predict(input1, input2, ..., inputN)
print(f"Prediction result: {result}")

# For more detailed information during prediction, use the debug parameter
result = model.predict(input1, input2, ..., inputN, debug=True)

# To control precision of calculations and output, use the precision parameter
result = model.predict(input1, input2, ..., inputN, precision=2)
```

## Example Usage

### Example 1: XOR Logic Gate
```python
from neural_network import NeuralNetwork

model = NeuralNetwork()
model.load_model("xor_model.txt")

# Test the XOR logic gate
print("Testing XOR logic gate model:")
print(f"0 XOR 0 = {model.predict(0, 0, precision=2)}")
print(f"0 XOR 1 = {model.predict(0, 1, precision=2)}")
print(f"1 XOR 0 = {model.predict(1, 0, precision=2)}")
print(f"1 XOR 1 = {model.predict(1, 1, precision=2)}")
```

### Example 2: Using the debug mode
```python
from neural_network import NeuralNetwork

model = NeuralNetwork()
model.load_model("my_model.txt")

# Display detailed information about the computation
result = model.predict(1, 0, debug=True, precision=3)
print(f"Final result: {result}")
```

## API Reference

### `NeuralNetwork` Class

#### Methods
- `__init__()` - Initialize a new neural network
- `load_model(filename)` - Load a model from a text file
- `predict(*inputs, debug=False, precision=None)` - Make predictions with the model
- `save_model(filename)` - Save the current model to a text file
- `__str__()` - Get a string representation of the model architecture

#### Parameters
- `filename` - Path to the model file
- `*inputs` - Input values for prediction (must match input layer size)
- `debug` - Boolean flag to print detailed computation information
- `precision` - Number of decimal places for rounding (None for no rounding)

## Model File Format

The model file format is a simple text file:
- First line: Space-separated integers representing the sizes of each layer
- Subsequent lines: Space-separated floats representing weights and biases for each layer

Example:
```
2 3 1
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
0.01 0.02 0.03 0.04
```

## License

MIT
