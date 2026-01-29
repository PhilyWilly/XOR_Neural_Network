from layer import Layer


class Neural_Network:
    """XOR Neural Network implementation."""
    
    def __init__(self):
        self.input = []
        self.layers = [
            Layer(2, 4, activation_function='relu'),
            Layer(4, 2, activation_function='relu'),
            Layer(2, 1, activation_function='sigmoid')
        ]
    
    @property
    def layer_count(self):
        """Return the number of layers."""
        return len(self.layers)
    
    def get_neuron_count(self, layer_index):
        """
        Get number of neurons in a specific layer.
        
        Args:
            layer_index: Layer index (0 is input layer)
            
        Returns:
            Number of neurons in the layer
        """
        if layer_index == 0:
            return len(self.input)
        return len(self.layers[layer_index - 1].neurons)
    
    def get_previous_neuron_count(self, layer_index, neuron_index):
        """
        Get number of weights for a specific neuron.
        
        Args:
            layer_index: Layer index
            neuron_index: Neuron index within the layer
            
        Returns:
            Number of weights (previous layer neurons)
        """
        if layer_index == 0:
            return None
        return len(self.layers[layer_index - 1].neurons[neuron_index].weights)
    
    def get_weight(self, layer_index, neuron_index, weight_index):
        """
        Get a specific weight value.
        
        Args:
            layer_index: Layer index
            neuron_index: Neuron index within the layer
            weight_index: Weight index within the neuron
            
        Returns:
            Weight value or None for input layer
        """
        if layer_index == 0:
            return None
        return self.layers[layer_index - 1].neurons[neuron_index].weights[weight_index]
    
    def get_neuron_value(self, layer_index, neuron_index):
        """
        Get a specific neuron's value.
        
        Args:
            layer_index: Layer index (0 is input layer)
            neuron_index: Neuron index within the layer
            
        Returns:
            Neuron value
        """
        if layer_index == 0:
            return self.input[neuron_index]
        return self.layers[layer_index - 1].neurons[neuron_index].value
    
    def forward(self, input_data):
        """
        Forward pass through the network.
        
        Args:
            input_data: Input values for the network
            
        Returns:
            Output values from the final layer
        """
        self.input = list(input_data)
        current_values = self.input
        
        for layer in self.layers:
            current_values = layer.forward_layer(current_values)
        return current_values

    def get_action(self):
        """
        Get output values from the network.
        
        Returns:
            List of output values from final layer
        """
        output_layer_index = len(self.layers)
        return [
            self.get_neuron_value(output_layer_index, neuron_idx)
            for neuron_idx in range(self.get_neuron_count(output_layer_index))
        ]

    def backpropagation(self, correct_values):
        """
        Backpropagation to update weights and biases.
        
        Args:
            correct_values: Expected output values
        """
        # Calculate error for output layer
        output_layer = self.layers[-1]
        error = [
            correct_val - neuron.value
            for correct_val, neuron in zip(correct_values, output_layer.neurons)
        ]

        # Backpropagate through all layers in reverse order
        for i in range(len(self.layers) - 1, -1, -1):
            previous_layer = self.layers[i - 1] if i > 0 else self.input
            next_layer = self.layers[i + 1] if i < len(self.layers) - 1 else None
            
            error = self.layers[i].backpropagation(
                error,
                next_layer=next_layer,
                previous_layer=previous_layer
            )
            print(f"{error=}")

    def __str__(self):
        result = [f"Input: {self.input}\n"]
        result.extend(
            f"Layer {i}: \n{layer}\n"
            for i, layer in enumerate(self.layers)
        )
        return ''.join(result)


def run():
    """Main function to create and initialize the neural network."""
    nn = Neural_Network()


if __name__ == "__main__":
    run()