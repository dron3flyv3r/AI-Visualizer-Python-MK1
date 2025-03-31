"""
Examples demonstrating how to use the Neural Network interface
with models trained on aiviz.kasperlarsen.tech
"""

from neural_network import NeuralNetwork

def xor_example():
    """Example of using a model trained to mimic an XOR gate"""
    print("\n=== XOR Gate Example ===")
    model = NeuralNetwork()
    try:
        model.load_model("model.txt")  # Replace with your XOR model file
        print(model)
        
        print("XOR Logic Gate Predictions:")
        print(f"0 XOR 0 = {model.predict(0, 0, precision=2)[0]}")
        print(f"0 XOR 1 = {model.predict(0, 1, precision=2)[0]}")
        print(f"1 XOR 0 = {model.predict(1, 0, precision=2)[0]}")
        print(f"1 XOR 1 = {model.predict(1, 1, precision=2)[0]}")
    except Exception as e:
        print(f"Error loading or using XOR model: {e}")

def debug_mode_example():
    """Example demonstrating the debug mode for detailed computation information"""
    print("\n=== Debug Mode Example ===")
    model = NeuralNetwork()
    try:
        model.load_model("model.txt")  # Replace with your model file
        print("Making prediction with debug mode enabled:")
        result = model.predict(1, 0, debug=True, precision=3)
        print(f"Final result: {result}")
    except Exception as e:
        print(f"Error in debug mode example: {e}")

def precision_example():
    """Example showing the effect of different precision settings"""
    print("\n=== Precision Control Example ===")
    model = NeuralNetwork()
    try:
        model.load_model("model.txt")  # Replace with your model file
        input_values = [1, 0]
        
        print(f"Input: {input_values}")
        print(f"No rounding: {model.predict(*input_values)}")
        print(f"2 decimal places: {model.predict(*input_values, precision=2)}")
        print(f"4 decimal places: {model.predict(*input_values, precision=4)}")
    except Exception as e:
        print(f"Error in precision example: {e}")

if __name__ == "__main__":
    print("AI Visualizer Python Interface - Examples")
    print("----------------------------------------")
    print("These examples demonstrate how to use neural network models")
    print("trained on aiviz.kasperlarsen.tech")
    
    try:
        xor_example()
        debug_mode_example()
        precision_example()
    except Exception as e:
        print(f"Error running examples: {e}")
    
    print("\nTo run your own models:")
    print("1. Train your model on aiviz.kasperlarsen.tech")
    print("2. Download the model as a .txt file")
    print("3. Load it using model = NeuralNetwork(); model.load_model('your_model.txt')")
    print("4. Make predictions with model.predict(input1, input2, ...)")
