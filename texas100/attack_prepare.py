# prepare_attack_data.py

import json
import numpy as np
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set device
device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# Load the saved attack splits and the dataset
with open("attack_splits.json", "r") as f:
    attack_splits = json.load(f)

dataset = np.load("./texas100.npz")
features = dataset['features']
labels = dataset['labels']
#for now, im just concat attack train and test indicies, i'm just doing train test pslit in the rain file.
member_indicies = attack_splits["attack_train_mem_indices"]+attack_splits["attack_test_mem_indices"]
non_member_indices = attack_splits["attack_train_non_indices"]+attack_splits["attack_test_non_indices"]
print(len(member_indicies))
print(len(non_member_indices))
member_data = torch.tensor(features[member_indicies], dtype=torch.float32)
member_labels = torch.tensor(labels[member_indicies], dtype=torch.long)
non_member_data = torch.tensor(features[non_member_indices], dtype=torch.float32)
non_member_labels = torch.tensor(labels[non_member_indices], dtype=torch.long)

global_model_snapshots = torch.load("client_4_global_snapshots_round_3.pth")
amplification_factor = 1000

# Define selected epochs for calculating bias deltas
selected_epochs = [5, 10, 20, 25, 30, 35, 45, 50, 60, 85]

# Define function to create model
def create_model():
    model =  nn.Sequential(
        nn.Linear(6169, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 100)
    )
    for param in model.parameters():
        param.requires_grad = True
    return model

# Step 2: Amplify the bias deltas
def amplify_bias_deltas(bias_deltas, amplification_factor):
    amplified_deltas = []
    for delta in bias_deltas:
        amplified_delta = np.exp(amplification_factor * delta) - 1
        amplified_deltas.append(amplified_delta)
    return amplified_deltas

def construct_attack_vectors(global_model_snapshots, selected_epochs, amplification_factor):
    attack_vectors = []
    labels = []
    combined_data = np.concatenate([non_member_data.numpy(), member_data.numpy()])

    for idx, feature in enumerate(combined_data):
        feature_tensor = torch.tensor(feature, dtype=torch.float32).to(device)
        bias_changes = []

        
        criterion = nn.CrossEntropyLoss()

        # Calculate bias changes for the given data point across selected epochs
        for i in range(len(selected_epochs)):
            
            epoch = selected_epochs[i]

            # Load model snapshot for the selected epoch
            model_state_dict = global_model_snapshots[f"epoch_{epoch}"]
            model = create_model()
            model.load_state_dict(model_state_dict)
            model = model.to(device)
            # Record bias before training
            bias_before = model[-1].bias.detach().cpu().clone().numpy()

            # Define optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Determine the correct label
            if idx < len(member_data):
                label_value = member_labels[idx]
            else:
                label_value = non_member_labels[idx - len(member_data)]
            label_scalar = torch.argmax(label_value).item()
            label_tensor = torch.tensor([label_scalar], dtype=torch.long).to(device)

            # Train model with the single data point for multiple epochs
            num_epochs = 5  # Set this to the desired number of epochs
           
            for epoch_i in range(num_epochs):
                optimizer.zero_grad()
                output = model(feature_tensor.unsqueeze(0))
                loss = criterion(output, label_tensor)
                loss.backward()
                optimizer.step()
                #print(bias_gradient)
            # Record bias after training (after all epochs)
            bias_after = model[-1].bias.detach().cpu().numpy()

            # Record bias after training
            #print(f"bias before == bias after: {np.array_equal(bias_before,bias_after)}")
            # Calculate bias change for the data point
            #print("Bias before update:", bias_before)
            #print("Bias after update:", model[-1].bias.detach().cpu().numpy())
            bias_change = bias_after - bias_before
            
            bias_changes.append(bias_change)
            
            # Calculate the delta of bias changes between consecutive snapshots, temporal
        bias_deltas = []
        #print(len(bias_changes))
        for i in range(len(bias_changes) - 1):
            delta = bias_changes[i + 1] - bias_changes[i]
            bias_deltas.append(delta)
        #print(len(bias_deltas))
        flattened_deltas = np.concatenate(bias_deltas)
        
        # Visualize ampliefiers

        # amplification_factors = [5, 10, 50, 100,200]

        # # Create a figure to plot bias changes for different amplification factors
        # plt.figure(figsize=(15, len(amplification_factors) * 4))

                
        # for idx, factor in enumerate(amplification_factors):
        #     # Amplify the bias changes using the given amplification factor
        #     amplified_deltas = amplify_bias_deltas(bias_changes, factor)
        #     amplified_deltas_array = np.concatenate(amplified_deltas).flatten()  # Flatten to get a 1D array for distribution

        #     # Plot the histogram for the current amplification factor
        #     plt.subplot(len(amplification_factors), 1, idx + 1)
        #     plt.hist(amplified_deltas_array, bins=30, alpha=0.7, color='b')
        #     plt.title(f"Histogram of Bias Changes (Amplification Factor = {factor})")
        #     plt.xlabel("Bias Change Value")
        #     plt.ylabel("Frequency")
        #plt.tight_layout()
        #plt.show()
        
        amplified_deltas = amplify_bias_deltas(bias_deltas,100)
        #print(bias_deltas[0])
        #print(amplified_deltas[0])
        #print("amplified : " , amplified_deltas)
        # Construct the attack vector by concatenating amplified deltas
        attack_vector = np.concatenate(amplified_deltas, axis=0)
        attack_vectors.append(attack_vector)
        #print(attack_vector)
        # Determine if the data point is a member or not
        is_member = 1 if idx < len(member_data) else 0
        labels.append(is_member)

    return np.array(attack_vectors), np.array(labels)

logger.info("Constructing attack vectors for each data record by calculating bias changes and deltas...")
attack_vectors, attack_labels = construct_attack_vectors(global_model_snapshots, selected_epochs, amplification_factor)

# Save the attack vectors and labels for further use
np.save("attack_vectors.npy", attack_vectors)
np.save("attack_labels.npy", attack_labels)
logger.info("Attack vectors and labels saved to attack_vectors.npy and attack_labels.npy")
