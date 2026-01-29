import matplotlib.pyplot as plt
from math import cos, sin, atan
from IPython import display


class Visualise:
    """Visualize the neural network structure and weights."""
    
    def __init__(self, neural_net):
        plt.figure()
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
        self.neural_net = neural_net
        num_layers = neural_net.layer_count + 1
        
        self.lines = []
        # Draw weights
        for layer_idx in range(num_layers):
            self.lines.append([])
            num_neurons = neural_net.get_neuron_count(layer_idx)
            
            if layer_idx == 0:
                for _ in range(num_neurons):
                    self.lines[layer_idx].append([])
                continue
            
            prev_num_neurons = neural_net.get_neuron_count(layer_idx - 1)

            for neuron_idx in range(num_neurons):
                self.lines[layer_idx].append([])
                
                neuron_x = layer_idx / num_layers * 700
                prev_neuron_x = (layer_idx - 1) / num_layers * 700
                neuron_y = 400 - (neuron_idx / num_neurons * 400 + 400 / num_neurons / 2)
                
                for weight_idx in range(neural_net.get_previous_neuron_count(layer_idx, neuron_idx)):
                    prev_neuron_y = 400 - (weight_idx / prev_num_neurons * 400 + 400 / prev_num_neurons / 2)
                    line = plt.Line2D(
                        (neuron_x, prev_neuron_x),
                        (neuron_y, prev_neuron_y),
                        zorder=1
                    )
                    self.lines[layer_idx][neuron_idx].append(line)
                    plt.gca().add_line(line)
        
        self.circles = []
        # Draw neurons
        for layer_idx in range(num_layers):
            self.circles.append([])
            num_neurons = neural_net.get_neuron_count(layer_idx)
            
            for neuron_idx in range(num_neurons):
                x_pos = layer_idx / num_layers * 700
                y_pos = 400 - (neuron_idx / num_neurons * 400 + 400 / num_neurons / 2)
                circle = plt.Circle(
                    (x_pos, y_pos),
                    radius=16,
                    fill=True,
                    zorder=2,
                    linewidth=1,
                    facecolor='w',
                    edgecolor='k'
                )
                self.circles[layer_idx].append(circle)
                plt.gca().add_patch(circle)

        plt.axis('scaled')
        plt.axis('off')
        plt.title('Neural network', fontsize=15)
    
    def update(self, neural_network):
        """
        Update the visualization with current network state.
        
        Args:
            neural_network: Neural network to visualize
        """
        self.neural_net = neural_network

        for layer_idx in range(len(self.lines)):
            for neuron_idx in range(len(self.lines[layer_idx])):
                # Set circle color based on neuron value
                neuron_value = self.neural_net.get_neuron_value(layer_idx, neuron_idx)
                neuron_value = max(0, min(1.0, neuron_value))
                neuron_value = 1 - neuron_value
                neuron_value = int(255 * neuron_value)
                color_neuron = f'#{neuron_value:02X}{255:02X}{neuron_value:02X}'
                
                self.circles[layer_idx][neuron_idx].set_facecolor(color_neuron)
                
                # Update weight line colors
                for weight_idx in range(len(self.lines[layer_idx][neuron_idx])):
                    weight_value = self.neural_net.get_weight(layer_idx, neuron_idx, weight_idx)
                    color = self._interpolate_color(weight_value)
                    self.lines[layer_idx][neuron_idx][weight_idx].set_color(color)
        
        plt.show(block=False)
        plt.pause(0.1)

    def _interpolate_color(self, value):
        """
        Interpolate color based on weight value.
        
        Args:
            value: Weight value (between -1 and 1)
            
        Returns:
            Hex color string
        """
        # Clamp value between -1.0 and 1.0
        value = max(-1.0, min(1.0, value))

        # Calculate color components (positive=green, negative=red)
        red = int(-255 * value) if value < 0 else 0
        green = int(255 * value) if value > 0 else 0
        blue = 0

        return f'#{red:02X}{green:02X}{blue:02X}'
    



        

    