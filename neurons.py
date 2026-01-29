import random
import math

LEARNING_RATE = 0.1


class Neuron:
    """A single neuron in a neural network layer."""
    
    # Activation function mappings
    _ACTIVATION_FUNCTIONS = {
        'sigmoid': lambda x: 1 / (1 + math.exp(-x)),
        'relu': lambda x: max(0, x),
        'vanilla': lambda x: x,
    }
    
    _DERIVATIVE_FUNCTIONS = {
        'sigmoid': lambda x: x * (1 - x),
        'relu': lambda x: 1 if x > 0 else 0,
        'vanilla': lambda x: 1,
    }
    
    def __init__(self, neurons_in_previous_layer, index=None, activation_function='sigmoid'):
        self.value = 0
        self.raw_value = 0
        self.activation_function = activation_function
        self.index = index
        self.bias = 0
        self.weights = [random.uniform(-1, 1) for _ in range(neurons_in_previous_layer)]
        
        if activation_function not in self._ACTIVATION_FUNCTIONS:
            raise ValueError(f"Invalid activation function: {activation_function}")
    
    def activate(self, x):
        """Apply activation function to input value."""
        self.value = self._ACTIVATION_FUNCTIONS[self.activation_function](x)
    
    def derivative(self, x):
        """Calculate derivative of activation function."""
        return self._DERIVATIVE_FUNCTIONS[self.activation_function](x)

    def forward_neuron(self, inputs, apply_activation=True):
        """
        Forward pass: compute weighted sum of inputs plus bias.
        
        Args:
            inputs: List of input values from previous layer
            apply_activation: Whether to apply activation function
            
        Returns:
            Computed neuron value
        """
        weighted_sum = sum(inp * weight for inp, weight in zip(inputs, self.weights))
        self.raw_value = weighted_sum + self.bias

        if apply_activation:
            self.activate(self.raw_value)
        else:
            self.value = self.raw_value
        return self.value
        
    def backpropagation(self, correct_values, previous_layer, next_layer):
        """
        Backpropagation: update weights and bias based on error.
        
        Args:
            correct_values: Error values from next layer
            previous_layer: Previous layer or input values
            next_layer: Next layer (None for output layer)
            
        Returns:
            Gradient value to pass to previous layer
        """
        if previous_layer is None:
            return
        
        gradient = 0
        for i, correct_value in enumerate(correct_values):
            if next_layer is None:
                gradient += correct_value * self.derivative(self.value)
            else:
                gradient += correct_value * self.derivative(self.value) * next_layer.neurons[i].weights[self.index]
        
        self.bias += LEARNING_RATE * gradient
        
        for i, weight in enumerate(self.weights):
            if isinstance(previous_layer, list):
                input_neuron_value = previous_layer[i]
            else:
                input_neuron_value = previous_layer.neurons[i].value
            
            delta_weight = LEARNING_RATE * gradient * input_neuron_value
            self.weights[i] += delta_weight
        
        return gradient
    
    def __str__(self):
        weight_strings = [f"\t\tWeight {i}: {w}\n" for i, w in enumerate(self.weights)]
        return ''.join(weight_strings)

   
