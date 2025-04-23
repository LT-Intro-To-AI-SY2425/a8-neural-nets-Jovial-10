from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

# =======================
# Function to Monitor Convergence
# =======================
def monitor_convergence(network, training_data, max_iterations=10000, print_interval=100, error_threshold=0.0005):
    prev_error = float('inf')
    for iteration in range(max_iterations):
        network.train(training_data, iterations=1)  # Train for 1 iteration at a time
        if iteration % print_interval == 0:
            error = network.calculate_error(training_data)
            error_change = abs(prev_error - error)
            print(f"Iteration {iteration}: Error = {error:.6f}")
            if error_change < error_threshold:  # Check if convergence criteria are met
                print("Convergence reached!")
                break
            prev_error = error

# =======================
# Function to Evaluate Network
# =======================
def evaluate_network(network, training_data):
    for input_data, expected_output in training_data:
        predicted_output = network.predict(input_data)
        print(f"Input: {input_data}, Predicted: {predicted_output}, Expected: {expected_output}")

# =======================
# Create and Train Network with 2 Hidden Nodes
# =======================
print("<<<<<<<<<<<<<< XOR with 2 Hidden Nodes >>>>>>>>>>>>>>\n")

network_2_hidden = NeuralNet(input_nodes=2, hidden_nodes=2, output_nodes=1)

print("Monitoring Convergence for Network with 2 Hidden Nodes...")
monitor_convergence(network_2_hidden, training_data)

print("\nEvaluating Network with 2 Hidden Nodes...")
evaluate_network(network_2_hidden, training_data)

# =======================
# Create and Train Network with 8 Hidden Nodes
# =======================
print("\n<<<<<<<<<<<<<< XOR with 8 Hidden Nodes >>>>>>>>>>>>>>\n")

network_8_hidden = NeuralNet(input_nodes=2, hidden_nodes=8, output_nodes=1)

print("Monitoring Convergence for Network with 8 Hidden Nodes...")
monitor_convergence(network_8_hidden, training_data)

print("\nEvaluating Network with 8 Hidden Nodes...")
evaluate_network(network_8_hidden, training_data)

# =======================
# Create and Train Network with 1 Hidden Node
# =======================
print("\n<<<<<<<<<<<<<< XOR with 1 Hidden Node >>>>>>>>>>>>>>\n")

network_1_hidden = NeuralNet(input_nodes=2, hidden_nodes=1, output_nodes=1)

print("Monitoring Convergence for Network with 1 Hidden Node...")
monitor_convergence(network_1_hidden, training_data)

print("\nEvaluating Network with 1 Hidden Node...")
evaluate_network(network_1_hidden, training_data)


