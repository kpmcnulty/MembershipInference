import json
import numpy as np

# Configuration
selected_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num_clients = 5
em = 10

# Load the ground truth sources
ground_truth_sources = np.load("ground_truth_sources.npy")

# Load bias_deltas from the JSON file
def load_bias_deltas(file_path):
    with open(file_path, "r") as f:
        return {int(rnd): {int(client): [np.array(delta) for delta in deltas]
                           for client, deltas in clients.items()}
                for rnd, clients in json.load(f).items()}

# Classify sources by minimum bias
def classify_by_min_bias(bias_deltas, record_idx, selected_epochs, num_clients):
    min_bias_counts = {client_idx: 0 for client_idx in range(num_clients)}
    for rnd in selected_epochs:
        biases_for_record = [
            bias_deltas[rnd][client_idx][record_idx] for client_idx in range(num_clients)
        ]
        min_bias_client = np.argmin([np.sum(np.abs(bias)) for bias in biases_for_record])
        min_bias_counts[min_bias_client] += 1
    return max(min_bias_counts, key=min_bias_counts.get)

# Classify sources by max bias with amplification
def classify_by_max_bias(bias_deltas, record_idx, selected_epochs, num_clients, em):
    max_bias_counts = {client_idx: 0 for client_idx in range(num_clients)}
    for rnd in selected_epochs:
        biases_for_record = [
            bias_deltas[rnd][client_idx][record_idx] for client_idx in range(num_clients)
        ]
        amplified_biases = [
            np.exp(em * np.abs(bias)) / np.sum(np.exp(em * np.abs(biases_for_record)), axis=0)
            for bias in biases_for_record
        ]
        for node_idx in range(len(amplified_biases[0])):
            max_bias_client = np.argmax(
                [amplified_biases[client_idx][node_idx] for client_idx in range(num_clients)]
            )
            max_bias_counts[max_bias_client] += 1
    return max(max_bias_counts, key=max_bias_counts.get)

# Main execution
bias_deltas = load_bias_deltas("bias_deltas.json")

predictions_min = []
predictions_max = []

for record_idx in range(len(ground_truth_sources)):
    inferred_min_source = classify_by_min_bias(bias_deltas, record_idx, selected_epochs, num_clients)
    inferred_max_source = classify_by_max_bias(bias_deltas, record_idx, selected_epochs, num_clients, em)

    predictions_min.append(inferred_min_source)
    predictions_max.append(inferred_max_source)

# Calculate accuracy
correct_min = np.sum(np.array(predictions_min) == ground_truth_sources)
accuracy_min = correct_min / len(ground_truth_sources)

correct_max = np.sum(np.array(predictions_max) == ground_truth_sources)
accuracy_max = correct_max / len(ground_truth_sources)

# Print results
print(f"Attack Accuracy (minimum bias): {accuracy_min * 100:.2f}% ({correct_min}/{len(ground_truth_sources)} correct)")
print(f"Attack Accuracy (max bias): {accuracy_max * 100:.2f}% ({correct_max}/{len(ground_truth_sources)} correct)")
