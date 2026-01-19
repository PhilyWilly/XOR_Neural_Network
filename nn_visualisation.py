import matplotlib.pyplot as plt
from math import cos, sin, atan
from IPython import display


class Visualise:
    def __init__(self, neural_net) -> None:
        plt.figure()
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
        self.neural_net = neural_net

        n_layer = self.neural_net.layer_count()+1
        
        self.line = []
        # Draw weights
        for layer in range(n_layer):
            self.line.append([])
            n_neurons = n_neurons = self.neural_net.n_neuron(layer)
            if layer == 0:
                for neuron in range(n_neurons):
                    self.line[layer].append([])
                continue
            prev_n_neurons = self.neural_net.n_neuron(layer-1)

            for neuron in range(n_neurons):
                self.line[layer].append([])
                
                neuron_x = layer/n_layer*700
                prev_neuron_x = (layer-1)/n_layer*700
                neuron_y = 400 - (neuron/n_neurons*400+400/n_neurons/2)
                for weight in range(self.neural_net.previous_n_neuron(layer,neuron)):                 
                    self.line[layer][neuron].append(plt.Line2D((neuron_x, prev_neuron_x), (neuron_y, 400 - (weight/prev_n_neurons*400+400/prev_n_neurons/2)), zorder=1))
                    plt.gca().add_line(self.line[layer][neuron][weight])

        
        self.circle = []
        # Draw neurons
        for layer in range(n_layer):
            self.circle.append([])
            if layer == 0:
                input = self.neural_net.get_input()
                n_neurons = len(input)
                for neuron in range(n_neurons):
                    self.circle[layer].append(plt.Circle((layer/n_layer*700, 400 - (neuron/n_neurons*400+400/n_neurons/2)), radius=16, fill=True, zorder=2,linewidth=1, facecolor='w', edgecolor='k'))
                    plt.gca().add_patch(self.circle[layer][neuron])
            else:
                n_neurons = self.neural_net.n_neuron(layer)
                for neuron in range(n_neurons):
                    self.circle[layer].append(plt.Circle((layer/n_layer*700, 400 - (neuron/n_neurons*400+400/n_neurons/2)), radius=16, fill=True, zorder=2,linewidth=1, facecolor='w', edgecolor='k'))
                    plt.gca().add_patch(self.circle[layer][neuron])
        

        plt.axis('scaled')
        plt.axis('off')
        plt.title('Neural network', fontsize=15)
        #self.update(self.neural_net)
        
    
    def update(self, neural_network):
        self.neural_net = neural_network

        for layer in range(len(self.line)):
            for neuron in range(len(self.line[layer])):
                # Set circle color according to the value in that neuron
                
                neuron_value = self.neural_net.get_neuron_value(layer, neuron)
                neuron_value = max(0, min(1.0, neuron_value))
                neuron_value = -neuron_value + 1
                neuron_value = int(255*neuron_value)
                color_neuron = f'#{(neuron_value):02X}{255:02X}{(neuron_value):02X}'
                
                self.circle[layer][neuron].set_facecolor(color_neuron)
                
                for weight in range(len(self.line[layer][neuron])):
                    value = self.neural_net.get_weight(layer, neuron, weight)
                    color = self.interpolate_color(value)
                    self.line[layer][neuron][weight].set_color(color)
        
        plt.show(block=False)
        plt.pause(0.1)
        

    def interpolate_color(self, value):
        # Ensure value is clamped between -1.0 and 1.0
        value = max(-1.0, min(1.0, value))

        # Normalize value to the range [0, 1] for easier interpolation
        green = 0
        red = 0
        if value > 0:
            green = int(255 * value)
        else:
            red = int(-255 * value)
        blue = 0  # Constant for both colors

        # Return the interpolated color in hex format
        return f'#{red:02X}{green:02X}{blue:02X}'
    



        

    