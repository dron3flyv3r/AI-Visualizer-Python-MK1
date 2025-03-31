import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.weights = []  # List of weight matrices
        self.biases = []   # List of bias vectors
        self.layer_sizes = []  # List of layer sizes

    def sigmoid(self, x):
        """Sigmoid activation function with numerically stable clipping"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def load_model(self, filename):
        """Load model configuration from a text file"""
        with open(filename, 'r') as f:
            # Read all lines and remove any empty lines
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # First line contains layer sizes
            self.layer_sizes = [int(x) for x in lines[0].split()]
            
            # Initialize lists for weights and biases
            self.weights = []
            self.biases = []
            
            # Process each layer's parameters (weights and biases)
            for i in range(len(self.layer_sizes) - 1):
                if i + 1 >= len(lines):
                    raise ValueError(f"Missing parameters for layer {i}")
                
                # Get all parameters for this layer
                params = [float(x) for x in lines[i + 1].split()]
                
                # Calculate number of weights for this layer
                n_weights = self.layer_sizes[i] * self.layer_sizes[i + 1]
                if len(params) != n_weights + self.layer_sizes[i + 1]:
                    raise ValueError(f"Incorrect number of parameters for layer {i}")
                
                # Extract and reshape weights
                weights = np.array(params[:n_weights])
                weights = weights.reshape((self.layer_sizes[i], self.layer_sizes[i + 1]))
                self.weights.append(weights)
                
                # Extract biases for the next layer
                biases = np.array(params[n_weights:])
                self.biases.append(biases)

    def predict(self, *inputs, debug=False, precision=None):
        """Make a prediction with the model using explicit for loops
        Args:
            *inputs: Variable number of input values
            debug: Whether to print debug information
            precision: Number of decimal places to round to (None for no rounding)
        Returns:
            numpy array: Model predictions
        """
        if len(inputs) != self.layer_sizes[0]:
            raise ValueError(f"Expected {self.layer_sizes[0]} inputs, got {len(inputs)}")
        
        # Convert inputs to numpy array
        current = np.array(inputs)
        debug_separator = None
        calculation = []
        
        if debug:
            debug_separator = "-" * 60
            print(f"\n{debug_separator}")
            print(f"NEURAL NETWORK FORWARD PROPAGATION")
            print(f"{debug_separator}")
            print(f"\n📊 INPUT LAYER (Layer 0):")
            for i, value in enumerate(current):
                print(f"  Neuron {i}: {value}")
            
        # Forward pass through each layer
        for layer_idx, (layer_weights, layer_biases) in enumerate(zip(self.weights, self.biases)):
            if precision is not None:
                current = np.round(current, precision)
                layer_weights = np.round(layer_weights, precision)
                layer_biases = np.round(layer_biases, precision)
            
            if debug:
                next_layer_type = "OUTPUT" if layer_idx == len(self.weights) - 1 else f"HIDDEN {layer_idx+1}"
                print(f"\n{debug_separator}")
                print(f"LAYER {layer_idx} → LAYER {layer_idx+1} ({next_layer_type})")
                print(f"{debug_separator}")
                
                print(f"\n📐 Weights:")
                for i in range(layer_weights.shape[0]):
                    print(f"  {[f'{w:.4f}' for w in layer_weights[i]]}")
                
                print(f"\n📏 Biases:")
                print(f"  {[f'{b:.4f}' for b in layer_biases]}")
            
            # Manual matrix multiplication and bias addition
            next_layer_size = layer_weights.shape[1]
            weighted_sums = np.zeros(next_layer_size)
            
            if debug:
                print(f"\n🧮 Calculation details:")
            
            # For each neuron in the next layer
            for j in range(next_layer_size):
                weighted_sum = 0
                
                if debug:
                    print(f"\n  Neuron {j} in Layer {layer_idx+1}:")
                    calculation = []
                
                # For each input from the current layer
                for i in range(len(current)):
                    contribution = current[i] * layer_weights[i, j]
                    weighted_sum += contribution
                    
                    if debug:
                        calculation.append(f"({current[i]:.4f} × {layer_weights[i, j]:.4f} = {contribution:.4f})")
                    
                    if precision is not None:
                        weighted_sum = round(weighted_sum, precision)
                
                # Add bias
                if debug:
                    bias_str = f"{layer_biases[j]:.4f}" if layer_biases[j] >= 0 else f"({layer_biases[j]:.4f})"
                    print("  " + " + ".join(calculation) + f" + {bias_str} (bias)")
                
                weighted_sum += layer_biases[j]
                weighted_sums[j] = weighted_sum if precision is None else round(weighted_sum, precision)
                
                if debug:
                    print(f"  Weighted sum = {weighted_sums[j]:.4f}")
                    sigmoid_result = self.sigmoid(weighted_sums[j])
                    print(f"  After sigmoid: {sigmoid_result:.4f}")
            
            # Apply activation function
            current = np.array([self.sigmoid(x) for x in weighted_sums])
            if precision is not None:
                current = np.round(current, precision)
            
            if debug:
                print(f"\n🔢 Layer {layer_idx+1} Output:")
                for i, value in enumerate(current):
                    print(f"  Neuron {i}: {value:.4f}")
        
        # Round the final results if precision is specified
        if precision is not None:
            current = np.round(current, precision)
            
        if debug:
            print(f"\n{debug_separator}")
            print(f"FINAL OUTPUT:")
            print(f"{debug_separator}")
            print(f"{current}")
            print(f"{debug_separator}\n")
            
        return current

    def save_model(self, filename):
        """Save the current model configuration to a file"""
        with open(filename, 'w') as f:
            # Write layer sizes
            f.write(' '.join(map(str, self.layer_sizes)) + '\n')
            
            # Write weights and biases for each layer
            for layer_idx in range(len(self.weights)):
                # Flatten weights and concatenate with biases
                flat_weights = self.weights[layer_idx].flatten()
                parameters = np.concatenate([flat_weights, self.biases[layer_idx]])
                f.write(' '.join(map(str, parameters)) + '\n')

    def __str__(self):
        """String representation of the model architecture with enhanced formatting"""
        # Header with network name
        separator = "=" * 60
        architecture = f"\n{separator}\n"
        architecture += f"                NEURAL NETWORK ARCHITECTURE\n"
        architecture += f"{separator}\n\n"
        
        # Basic architecture info in a clean format
        total_layers = len(self.layer_sizes)
        total_neurons = sum(self.layer_sizes)
        total_connections = sum(w.size for w in self.weights)
        
        architecture += f"│ NETWORK SUMMARY:\n"
        architecture += f"├─ Total layers: {total_layers} ({total_layers-2} hidden)\n"
        architecture += f"├─ Total neurons: {total_neurons}\n"
        architecture += f"└─ Total connections: {total_connections}\n\n"
        
        # Enhanced layer structure visualization
        architecture += f"NETWORK TOPOLOGY:\n\n"
        
        # Create a textual representation of layers
        layer_repr = []
        for i, size in enumerate(self.layer_sizes):
            if i == 0:
                layer_type = "INPUT LAYER"
            elif i == len(self.layer_sizes) - 1:
                layer_type = "OUTPUT LAYER"
            else:
                layer_type = f"HIDDEN LAYER {i}"
                
            layer_repr.append(f"[{layer_type}: {size} neurons]")
        
        # Connect layers with arrows
        architecture += "  " + " ──→ ".join(layer_repr) + "\n\n"
        
        # Add more detailed breakdown of layers
        architecture += "LAYER BREAKDOWN:\n"
        
        for i, size in enumerate(self.layer_sizes):
            if i == 0:
                architecture += f"  • Layer {i} (Input): {size} neurons\n"
            elif i == len(self.layer_sizes) - 1:
                architecture += f"  • Layer {i} (Output): {size} neurons\n"
            else:
                architecture += f"  • Layer {i} (Hidden): {size} neurons\n"
        
        architecture += "\n"
        
        # Details about each layer's weights and biases
        architecture += f"{separator}\n"
        architecture += "DETAILED PARAMETER INFORMATION:\n"
        architecture += f"{separator}\n\n"
        
        total_params = 0
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            total_params += w.size + b.size
            
            from_layer = "Input" if i == 0 else f"Hidden {i}"
            to_layer = "Output" if i == len(self.weights)-1 else f"Hidden {i+1}"
            
            architecture += f"Connection {i}: {from_layer} → {to_layer}\n"
            architecture += f"├─ Neurons: {self.layer_sizes[i]} → {self.layer_sizes[i+1]}\n"
            architecture += f"├─ Weight matrix: {w.shape[0]}×{w.shape[1]} = {w.size} parameters\n"
            architecture += f"├─ Bias vector: {b.shape[0]} parameters\n"
            architecture += f"├─ Weight stats: min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}, std={w.std():.4f}\n"
            architecture += f"└─ Bias stats: min={b.min():.4f}, max={b.max():.4f}, mean={b.mean():.4f}, std={b.std():.4f}\n\n"
        
        architecture += f"{separator}\n"
        architecture += f"Total trainable parameters: {total_params}\n"
        architecture += f"{separator}\n"
        
        return architecture


if __name__ == "__main__":
    # Example usage
    model = NeuralNetwork()
    
    try:
        model.load_model("model.txt")
        print("\nModel loaded successfully!")
        print(model)
        
        print(model.predict(1, 0, precision=2, debug=True))
        print(model.predict(1, 1, precision=2))
        print(model.predict(0, 1, precision=2))
        
        
        # Example predictions
        # n_inputs = model.layer_sizes[0]
        # test_inputs = [0.5] * n_inputs
        # print(f"\nTest prediction with inputs {test_inputs}:")
        # result = model.predict(*test_inputs)
        # print(f"Output: {result}")
        
        # if n_inputs == 2:  # Test with another example for 2-input networks
        #     print("\nTest prediction with inputs [1, 0]:")
        #     result = model.predict(1, 0, debug=True, precision=2)
        #     print(f"Output: {result}")
            
        #     # Check with exact values used in web version
        #     print("\nTest prediction with inputs [1, 1]:")
        #     result = model.predict(1, 1)
        #     print(f"Output: {result}")
            
    except Exception as e:
        print(f"Error: {e}")