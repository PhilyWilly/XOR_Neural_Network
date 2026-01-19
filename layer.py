from neurons import Neuron
import math

class Layer:
    def __init__(self, neurons_in_previous_layer, neurons_in_current_layer, activation_function = "sigmoid") -> None:
        self.nipl = neurons_in_previous_layer
        self.activation_function = activation_function
        self.neurons = []
        for i in range(neurons_in_current_layer):
            self.neurons.append(Neuron(self.nipl, index=i, activation_function=self.activation_function))

    def n_neuron(self):
        return len(self.neurons)
    
    def previous_neuron_count(self, neuron):
        return self.neurons[neuron].n_weight()
    
    def get_weight(self, neuron, weight):
        return self.neurons[neuron].get_weight(weight)
    
    def get_neuron_value(self, neuron):
        return self.neurons[neuron].get_value()
    
    # Calculate solution
    def set_neuron_value(self, neuron_index, value):
        self.neurons[neuron_index].set_value(value)

    def forward_layer(self, inputs):
        if self.activation_function != "soft-max":
            return [neuron.forward_neuron(inputs) for neuron in self.neurons]
            
        else:
            # Soft-max algorhythm
            neuron_sum = 0
            #Get the sum of all neurons
            for i in range(len(self.neurons)):
                self.neurons[i].forward_neuron(inputs)
                neuron_sum += math.e**self.neurons[i].get_value()
            
            # Change the value of each neuron
            for i in range(len(self.neurons)):
                neuron_value_before = math.e**self.neurons[i].get_value()
                new_neuron_value = neuron_value_before/neuron_sum
                #print(f"The sum is {neuron_sum}, the previous value was {neuron_value_before} and the new neuron value is {new_neuron_value}")
                self.neurons[i].set_value(new_neuron_value)
            
    # The Neuron addition from pprebvious lkawodjioaddaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    def backpropagation(self, correct_delta_values, next_layer=None, previous_layer=None):
        pass_to_next = []
        for neuron in self.neurons:
            pass_to_next.append(neuron.backpropagation(correct_delta_values, previous_layer=previous_layer, next_layer=next_layer))

        return pass_to_next
    
    def __str__(self):
        ret_str = ""
        for i, n in enumerate(self.neurons):
            ret_str += f"\tNeuron {i}: {n.value}\n\t\tBias: {n.bias}\n{n}\n"
        return ret_str


            
