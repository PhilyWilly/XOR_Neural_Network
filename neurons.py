import enum
import random
import math

LEARNING_RATE = 0.1

class Neuron:
    def __init__(self, neurons_in_previous_layer, index = None, activation_function = "sigmoid") -> None:
        self.value = 0
        self.raw_value = 0

        self.activation_function = activation_function
        self.index = index
        self.bias = 0
        self.weights = []
        for i in range(neurons_in_previous_layer):
            self.weights.append(random.uniform(-1,1))
    
    def n_weight(self):
        return len(self.weights)
    
    def get_weight(self, weight):
        return self.weights[weight]
    
    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value

    def activate(self, x):
        match self.activation_function:
            case "sigmoid":
                self.value = 1/(1 + math.exp(-x))
            case "relu":
                self.value = max(0, x)
            case "vanilla":
                self.value = x
            case _:
                raise Exception("No valid activation function!")

    def derivative(self, x):
        match self.activation_function:
            case "sigmoid":
                return x * (1 - x)
            case "relu":
                return 1 if x > 0 else 0
            case "vanilla":
                return 1
            case _:
                raise Exception("Error in activation function!")

    def forward_neuron(self, inputs, activation_function=True):
        # Sum of all neurons multiplied by the weights
        sum = 0
        for i in range(len(self.weights)):
            neuron_value = inputs[i]
            sum += neuron_value*self.weights[i]
        # Bias addition
        self.raw_value = sum + self.bias

        # Activation function
        if activation_function:
            self.value = self.activate(self.raw_value)
        else:
            self.value = self.raw_value
        return self.value    
        
    # Correct value is the output this neuron should have
    def backpropagation(self, correct_values, previous_layer, next_layer):
        if previous_layer is None:
            return
        
        gradient = 0
        for i, cor in enumerate(correct_values):
            if next_layer == None:
                gradient += cor * self.derivative(self.value)
            else:
                gradient += cor * self.derivative(self.value) * next_layer.get_weight(i, self.index)
        
        self.bias += LEARNING_RATE * gradient
        for i in range(len(self.weights)):
            if type(previous_layer) == list:
                input_neuron_value = previous_layer[i]
            else:
                input_neuron_value = previous_layer.get_neuron_value(i)

            delta_weight = LEARNING_RATE * gradient * input_neuron_value
            self.weights[i] += delta_weight
        
        return gradient
    
    def __str__(self):
        ret_str = ""
        for i, w in enumerate(self.weights):
            ret_str += f"\t\tWeight {i}: {w}\n"
        return ret_str

   
