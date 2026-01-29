from neural_network import Neural_Network
from nn_visualisation import Visualise
import time

# XOR truth table
XOR_TRUTH_TABLE = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 1,
    (1, 1): 0
}

# Expected outputs for XOR inputs
XOR_EXPECTED_OUTPUTS = [[0], [1], [1], [0]]

# Training parameters
EPOCHS = 50000
DISPLAY_INTERVAL = 1000
PRINT_INTERVAL = 10
PAUSE_INTERVAL = 10000
AVG_COST_SMOOTHING = 0.1


def calculate_cost(output_values, expected_values):
    """
    Calculate mean squared error cost.
    
    Args:
        output_values: Actual network output values
        expected_values: Expected output values
        
    Returns:
        Total squared error cost
    """
    return sum((output - expected) ** 2 for output, expected in zip(output_values, expected_values))


def main():
    """Main training loop for XOR neural network."""
    nn = Neural_Network()
    
    # Initialize neural network with one forward pass for visualization
    nn.forward((0, 0))
    visualiser = Visualise(nn)
    
    avg_cost = None
    
    for epoch in range(EPOCHS):
        for i, xor_input in enumerate(XOR_TRUTH_TABLE):
            # Forward pass
            output = nn.forward(xor_input)
            
            # Update visualization periodically
            if epoch % DISPLAY_INTERVAL == 0:
                visualiser.update(nn)
                time.sleep(0.05)
            
            # Calculate cost
            cost = calculate_cost(output, XOR_EXPECTED_OUTPUTS[i])
            
            # Update average cost using exponential moving average
            if avg_cost is None:
                avg_cost = cost
            else:
                avg_cost = AVG_COST_SMOOTHING * cost + (1 - AVG_COST_SMOOTHING) * avg_cost
            
            # Print network state periodically
            if epoch % PAUSE_INTERVAL == 0:
                print(nn)
            
            # Backpropagation to learn
            nn.backpropagation(XOR_EXPECTED_OUTPUTS[i])
            
            # Print progress periodically
            if epoch % PRINT_INTERVAL == 0:
                print(f"Epoch: {epoch:5d} | Avg cost: {avg_cost:.3f} | Input: {xor_input} | Output: {output} | Cost: {cost:.3f}")
            
            # Pause for user inspection periodically
            if epoch % PAUSE_INTERVAL == 0:
                input("Press Enter to continue...")
    
    # Keep visualization window open
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()



