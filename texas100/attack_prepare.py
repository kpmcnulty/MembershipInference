import json
import numpy as np
import torch
import torch.nn as nn
import logging
import gc
import matplotlib.pyplot as plt

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the saved attack splits and the dataset
with open("attack_splits.json", "r") as f:
    attack_splits = json.load(f)

dataset = np.load("./texas100.npz")
features = dataset['features']
labels = dataset['labels']

# Retrieve member and non-member indices
member_indices = attack_splits["member_indices"]
#print(member_indices[:5])
non_member_indices = attack_splits["non_member_indices"]
#print(non_member_indices[:5])

# Move data to the correct device
member_data = torch.tensor(features[member_indices], dtype=torch.float32, device=device)
member_labels = torch.tensor(labels[member_indices], dtype=torch.long, device=device)
non_member_data = torch.tensor(features[non_member_indices], dtype=torch.float32, device=device)
non_member_labels = torch.tensor(labels[non_member_indices], dtype=torch.long, device=device)


# Path to the combined snapshots file
combined_snapshots_path = "combined_model_snapshots.pth"

# Amplification factor
amplification_factor = 100

# Define function to create model
def create_model():
    model = nn.Sequential(
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

# Function to amplify bias deltas
def amplify_bias_deltas(bias_deltas, amplification_factor):
    return [np.exp(amplification_factor * delta) - 1 for delta in bias_deltas]

def compute_attack_vector(batch_indices, combined_data, is_member_list, amplification_factor, selected_snapshots):
    attack_vectors = []
    labels = []
    all_bias_deltas = []

    # Define criterion
    criterion = nn.CrossEntropyLoss()

    # Iterate over each data point in the batch
    for idx in batch_indices:
        feature = combined_data[idx]
        feature_tensor = feature.unsqueeze(0).to(device)
        bias_changes = []

        # Loop through selected snapshots
        for snapshot_id, model_state_dict in selected_snapshots.items():
            model = create_model().to(device)
            model.load_state_dict(model_state_dict)
            # Record bias before training
            bias_before = model[-1].bias.detach().cpu().clone().numpy()

            # Define optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Determine the correct label
            label_value = member_labels[idx] if is_member_list[idx] else non_member_labels[idx - len(member_data)]
            label_scalar = torch.argmax(label_value).item()
            label_tensor = torch.tensor([label_scalar], dtype=torch.long, device=device)

            # Train model with the single data point for multiple epochs
            num_epochs = 10
            for _ in range(num_epochs):
                optimizer.zero_grad()
                output = model(feature_tensor)
                loss = criterion(output, label_tensor)
                loss.backward()
                optimizer.step()

            # Record bias after training
            bias_after = model[-1].bias.detach().cpu().numpy()

            # Calculate bias change
            bias_changes.append(bias_after - bias_before)

        # Calculate the delta of bias changes between consecutive snapshots
        bias_deltas = [bias_changes[i + 1] - bias_changes[i] for i in range(len(bias_changes) - 1)]
        all_bias_deltas.extend(bias_deltas)

        # Amplify the bias deltas and construct the attack vector
        amplified_deltas = amplify_bias_deltas(bias_deltas, amplification_factor)
        attack_vector = np.concatenate(amplified_deltas, axis=0)

        # Determine the label for the attack
        label = 1 if is_member_list[idx] else 0
        attack_vectors.append(attack_vector)
        labels.append(label)

    # Plot histograms for the whole batch
    #amplification_factors = [1, 5, 20, 50, 100]
    # plt.figure(figsize=(10, len(amplification_factors) * 3))

    # for idx, factor in enumerate(amplification_factors):
    #     amplified_deltas = amplify_bias_deltas(all_bias_deltas, factor)
    #     flattened_deltas = np.concatenate(amplified_deltas)  # Flatten for histogram plotting

    #     plt.subplot(len(amplification_factors), 1, idx + 1)
    #     plt.hist(flattened_deltas, bins=30, alpha=0.7, color='b')
    #     plt.title(f"Histogram of Bias Changes (Amplification Factor = {factor})")
    #     plt.xlabel("Bias Change Value")
    #     plt.ylabel("Frequency")

    # plt.tight_layout()
    # plt.show()

    return attack_vectors, labels


# Load the pre-combined snapshots
combined_snapshots = torch.load(combined_snapshots_path)
logger.info(f"Loaded combined snapshots for epochs: {list(combined_snapshots.keys())}")

# Prepare combined data and labels
combined_data = torch.cat([member_data, non_member_data])
is_member_list = [1] * len(member_data) + [0] * len(non_member_data)


# Batch configuration
batch_size = 64
num_batches = (len(combined_data) + batch_size - 1) // batch_size
batch_indices_list = [range(i * batch_size, min((i + 1) * batch_size, len(combined_data))) for i in range(num_batches)]

attack_vectors = []
labels = []

# Process each batch
for i, batch_indices in enumerate(batch_indices_list):
    logger.info(f"Processing batch {i + 1} / {num_batches}...")
    batch_attack_vectors, batch_labels = compute_attack_vector(
        batch_indices, combined_data, is_member_list, amplification_factor, combined_snapshots
    )
    attack_vectors.extend(batch_attack_vectors)
    labels.extend(batch_labels)
    gc.collect()

# Save the attack vectors and labels for further use
np.save("attack_vectors.npy", np.array(attack_vectors))
np.save("attack_labels.npy", np.array(labels))
logger.info("Attack vectors and labels saved to attack_vectors.npy and attack_labels.npy")
