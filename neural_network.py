from re import S
from layer import Layer


class Neural_Network:
    def __init__(self) -> None:
        self.input = []
        self.layer = []
        self.layer.append(Layer(2, 4, activation_function="relu"))
        self.layer.append(Layer(4, 2, activation_function="relu"))
        self.layer.append(Layer(2, 1, activation_function="sigmoid"))
    
    # For visualisation
    def layer_count(self):
        return len(self.layer)
    
    def n_neuron(self, layer):
        if layer == 0:
            return len(self.input)
        return self.layer[layer-1].n_neuron()
    
    def previous_n_neuron(self, layer, neuron):
        if layer == 0:
            return None
        return self.layer[layer-1].previous_neuron_count(neuron)
    
    def get_weight(self, layer, neuron, weight):
        if layer == 0:
            return None
        return self.layer[layer-1].get_weight(neuron, weight)
    
    def get_neuron_value(self, layer, neuron):
        if layer == 0:
            return self.input[neuron]
        return self.layer[layer-1].get_neuron_value(neuron)
    
    def get_input(self):
        return self.input
    
    # For Tensor calculations
    def forward(self, input):
        self.input = [n for n in input]
        current_neuron_list = input
        
        for layer in self.layer:
            current_neuron_list = [n for n in layer.forward_layer(current_neuron_list)]
        return current_neuron_list

    def get_action(self):
        action_list = []
        output_layer_index = len(self.layer)-1
        for neuron in range(self.n_neuron(output_layer_index)):
            action_list.append(self.get_neuron_value(output_layer_index, neuron))
        return action_list
    

    def backpropagation(self, correct_values):
        error = []
        for i, cor_val in enumerate(correct_values):
            value = self.layer[len(self.layer)-1].get_neuron_value(i)
            error.append(cor_val-value)

        for i in range(len(self.layer)-1, -1, -1):
            # print("Backpropagate layer: " + str(i))
            previous_layer = None
            if i != 0:
                previous_layer = self.layer[i-1]
            else:
                previous_layer = self.input

            next_layer = None
            if not i > self.layer_count()-2:
                next_layer = self.layer[i+1]
            error = [n for n in self.layer[i].backpropagation(error, next_layer=next_layer, previous_layer=previous_layer)]
            print(f"{error=}")

    def __str__(self):
        print("Stringify")
        ret_str = ""
        ret_str += f"Input: {self.input}\n"
        for i, l in enumerate(self.layer):
            # print(i)
            ret_str += f"Layer {i}: \n{l}\n"
        return ret_str 

    


def run():
    nn = Neural_Network()


if __name__ == "__main__":
    run()