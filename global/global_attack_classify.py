import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

num_epochs = 5 #### for filename stuff

def classify_source(attack_vector, amplification_factor):
    num_clients, num_epochs, num_nodes = attack_vector.shape
    client_wins = np.zeros(num_clients, dtype=int)
    
    # For each node in last layer
    for node_idx in range(num_nodes):
        node_max_client = None
        node_max_value = float('-inf')
        
        # For each client
        for client_idx in range(num_clients):
            # Get all epochs' bias values for this client and node
            bias_values = attack_vector[client_idx, :, node_idx]  # shape: (num_epochs,)
            
            # equation 14: normalize across epochs for this client-node combo
            amplified = np.exp(amplification_factor * bias_values)
            normalized = amplified / np.sum(amplified)  # Will sum to 1 across epochs for each node-client combo
            
            max_norm_value = np.max(normalized)
            
            # Track which client has highest normalized value for this node
            if max_norm_value > node_max_value:
                node_max_value = max_norm_value
                node_max_client = client_idx
        
        client_wins[node_max_client] += 1
        
    return np.argmax(client_wins)



# Load data
attack_vectors = np.load("global_attack_vectors5.npy")
attack_labels = np.load("global_attack_labels5.npy")
logger.info(f"Loaded attack vectors with shape {attack_vectors.shape}")

# Test different amplification factors
amplification_factors = [1,5,10,50,100,500,1000]

for factor in amplification_factors:
    logger.info(f"\nTesting amplification factor: {factor}")
    correct = 0
    total = 0
    
    # Test all samples
    for sample_idx in range(len(attack_vectors)):
        pred = classify_source(attack_vectors[sample_idx], factor)
        actual = attack_labels[sample_idx]
        correct += (pred == actual)
        total += 1
        
        
    
    # Final accuracy for this amplification factor
    accuracy = correct / total
    logger.info(f"Final accuracy for amplification factor {factor}: {accuracy:.3f}")


