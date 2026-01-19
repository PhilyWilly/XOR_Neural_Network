import random
import math

class Neuron:
    def __init__(self, neurons_in_previous_layer, index = None, activation_function = "sigmoid") -> None:
        self.value = 0
        self.raw_value = 0

        self.activation_function = activation_function
        self.next_layer = None
        self.index = index
        self.bias = 0
        self.weights = []
        for i in range(neurons_in_previous_layer):
            #self.weights.append(random.uniform(-(math.sqrt(6)/math.sqrt(neurons_in_previous_layer + neurons_in_next_layer)), math.sqrt(6)/math.sqrt(neurons_in_previous_layer + neurons_in_next_layer)))
            self.weights.append(random.uniform(-1,1))
            #print(self.weights[i])

    def next_layer_is(self, next_layer):
        self.next_layer = next_layer
    
    def n_weight(self):
        return len(self.weights)
    
    def get_weight(self, weight):
        return self.weights[weight]
    
    def get_value(self):
        return self.value
    
    def set_value(self, value):
        self.value = value

    def derivative(self, x):
        return x * (1 - x)

    def forward_neuron(self, inputs):
        # Sum of all neurons multiplied by the weights
        
        sum = 0
        for i in range(len(self.weights)):
            neuron_value = inputs[i]
            sum += neuron_value*self.weights[i]
        # Bias addition
        self.raw_value = sum + self.bias

        # Calculation
        if self.activation_function == "relu":
            if self.raw_value <= 0:
                self.value = 0
            else:
                self.value = self.raw_value
        
        elif self.activation_function == "sigmoid":
            #print("Sigma yes")
            sigmoid = 1/(1 + math.exp(-self.raw_value))
            #sigmoid = 1/(1 + math.e**-self.raw_value)
            #print(f"Before sigmoid: {value_before_calculation_mechanism} and after Simanoid: {sigmoid}")
            self.value = sigmoid

        elif self.activation_function == "vanilla":
            self.value = self.raw_value

        else:
            print("False activation function!")
        return self.value    
        
    # Correct value is the output this neuron should have
    def backpropagation(self, correct_values, previous_layer, next_layer):
        if previous_layer is None:
            return None
        learning_rate = 0.1

        gradient = 0
        for i, cor in enumerate(correct_values):
            if self.next_layer == None:
                gradient += cor * self.derivative(self.value)
            else:
                gradient += cor * self.derivative(self.value) * next_layer.get_weight(i, self.index)
        
        self.bias += learning_rate * gradient
        for i in range(len(self.weights)):
            if type(previous_layer) == list:
                input_neuron_value = previous_layer[i]
            else:
                input_neuron_value = previous_layer.get_neuron_value(i)

            delta_weight = learning_rate * gradient * input_neuron_value
            self.weights[i] += delta_weight
        
        return gradient

   
