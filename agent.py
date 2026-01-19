from neural_network import Neural_Network
from nn_visualisation import Visualise
import time

nn = Neural_Network()
average_cost = 0
epoch = 50000
batch = range(4)

XOR = {
    (0,0) : 0,
    (0,1) : 1,
    (1,0) : 1,
    (1,1) : 0
}
xor_correct_list = [0,1,1,0]

def get_cost(output_list, correct_label):
    cost_in_funtion = 0
    for output in output_list:
        cost_in_funtion += pow(output-correct_label, 2)
    
    return cost_in_funtion

def get_2d_cost(output_list, correct_list):
    cost_in_function = 0
    for out, cor in zip(output_list, correct_list):
        cost_in_function += pow(out - cor, 2)
    return cost_in_function

avg_cost = None
cost_n = 1

nn.forward((0,0)) # Needed a one time input for visualization
visualiser = Visualise(nn)
for ep in range(epoch): # New epoch
    for i, xor_state in enumerate(XOR): # New state in epoch
        action = nn.forward(xor_state) # Calculate how the current nn displays the XOR

        if ep % 1000 == 0: # Display the nn
            visualiser.update(nn)
            time.sleep(0.05)

        cost = get_cost(action, xor_correct_list[i])
        
        # Calc the avg cost
        if avg_cost is None:
            avg_cost = cost
        else:
            avg_cost = 0.1 * cost + (1-0.1) * avg_cost

        if ep % 10 == 0:
            print(f"Old  Epoch: {ep} Avrg cost: {avg_cost:.3f}    Input: {xor_state}    Action: {action}      Cost: {cost}")

        nn.backpropagation(xor_correct_list[i]) # Do the backpropagation (here is the part where the algorythm learns (this alg drove me crazy aaaaaaa))
        
        if ep % 10000 == 0: 
            print(nn)
            input("Press Enter to continue...")

while True:
    # To keep the neural net displayed
    time.sleep(0.1)
    



