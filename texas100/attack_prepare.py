# prepare_attack_data.py

import json
import numpy as np
import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='prepare_data.log', encoding='utf-8', level=logging.INFO)


# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the saved attack splits and the dataset
with open("attack_splits.json", "r") as f:
    attack_splits = json.load(f)

dataset = np.load("./texas100.npz")
features = dataset['features']
global_model_snapshots = torch.load("client_4_global_snapshots.pth")

# Define selected epochs for calculating bias deltas
selected_epochs = [5, 10, 20, 25, 30, 35, 45, 50, 60, 85]

def create_model():
    return nn.Sequential(
        nn.Linear(6169, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 100)
    )

def get_last_layer_biases(model_state_dict, data):
    model = create_model()
    model.load_state_dict(model_state_dict)
    model.eval()
    biases = []
    with torch.no_grad():
        for X in data:
            _ = model(X)
            last_layer_bias = model[-1].bias.detach().cpu().numpy()
            biases.append(last_layer_bias)
    return np.array(biases)

def calculate_bias_deltas(model_snapshots, data):
    bias_deltas = []
    for i in range(len(selected_epochs) - 1):
        model1 = model_snapshots[f"epoch_{selected_epochs[i]}"]
        model2 = model_snapshots[f"epoch_{selected_epochs[i + 1]}"]
        biases_epoch1 = get_last_layer_biases(model1, data)
        biases_epoch2 = get_last_layer_biases(model2, data)
        delta_bias = biases_epoch2 - biases_epoch1
        bias_deltas.append(delta_bias)
    return np.array(bias_deltas).reshape(-1, 100)

# Prepare member and non-member data
member_data = torch.tensor(features[attack_splits["attack_train_mem_indices"]], dtype=torch.float32)
non_member_data = torch.tensor(features[attack_splits["attack_train_non_indices"]], dtype=torch.float32)

# Calculate bias deltas
member_bias_deltas = calculate_bias_deltas(global_model_snapshots, member_data)
non_member_bias_deltas = calculate_bias_deltas(global_model_snapshots, non_member_data)

# Combine to form attack dataset
X_attack = np.concatenate([member_bias_deltas, non_member_bias_deltas], axis=0)
y_attack = np.concatenate([np.ones(member_bias_deltas.shape[0]), np.zeros(non_member_bias_deltas.shape[0])])

# Save attack dataset
np.savez("prepared_attack_data.npz", X_attack=X_attack, y_attack=y_attack)
logger.info("Attack data prepared and saved.")
