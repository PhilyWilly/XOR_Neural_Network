from neurons import Neuron
import math


class Layer:
    """A layer of neurons in a neural network."""
    
    def __init__(self, neurons_in_previous_layer, neurons_in_current_layer, activation_function='sigmoid'):
        self.neurons_in_prev_layer = neurons_in_previous_layer
        self.activation_function = activation_function
        self.neurons = [
            Neuron(self.neurons_in_prev_layer, index=i, activation_function=self.activation_function)
            for i in range(neurons_in_current_layer)
        ]

    def forward_layer(self, inputs):
        """
        Forward pass through the layer.
        
        Args:
            inputs: Input values from previous layer
            
        Returns:
            List of output values from this layer
        """
        if self.activation_function != 'soft-max':
            return [neuron.forward_neuron(inputs) for neuron in self.neurons]
        
        # Softmax activation
        for neuron in self.neurons:
            neuron.forward_neuron(inputs, apply_activation=False)
        
        # Calculate exponentials and sum
        exp_values = [math.exp(neuron.value) for neuron in self.neurons]
        exp_sum = sum(exp_values)
        
        # Normalize by the sum to get softmax output
        for neuron, exp_value in zip(self.neurons, exp_values):
            neuron.value = exp_value / exp_sum
            
        return [neuron.value for neuron in self.neurons]
            
    def backpropagation(self, correct_delta_values, next_layer=None, previous_layer=None):
        """
        Backpropagation through the layer.
        
        Args:
            correct_delta_values: Error values from next layer
            next_layer: Next layer (None for output layer)
            previous_layer: Previous layer or input values
            
        Returns:
            Gradient values to pass to previous layer
        """
        return [
            neuron.backpropagation(
                correct_delta_values, 
                previous_layer=previous_layer, 
                next_layer=next_layer
            )
            for neuron in self.neurons
        ]
    
    def __str__(self):
        neuron_strings = [
            f"\tNeuron {i}: {neuron.value}\n\t\tBias: {neuron.bias}\n{neuron}\n"
            for i, neuron in enumerate(self.neurons)
        ]
        return ''.join(neuron_strings)

