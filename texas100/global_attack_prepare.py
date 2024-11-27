import os
import json
import torch
import numpy as np
from torch import nn

# Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_dir = "models"
selected_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
num_clients = 5

# Dataset
dataset = np.load("./texas100.npz")
indices = np.load("global_train_indices.npy")
train_features = dataset["features"][indices]
train_labels = dataset["labels"][indices]

# Load the model architecture
def create_model():
    return nn.Sequential(
        nn.Linear(6169, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 100),
        nn.Softmax(dim=1),
    )

# Compute bias deltas for a single record
def compute_bias_delta(model, record_features, record_label):
    bias_before = model[-2].bias.detach().cpu().clone().numpy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for _ in range(5):  # Simulate training for num_epochs=5
        optimizer.zero_grad()
        output = model(record_features)
        loss = criterion(output, record_label)
        loss.backward()
        optimizer.step()

    bias_after = model[-2].bias.detach().cpu().numpy()
    return bias_after - bias_before

# Collect bias deltas
def construct_bias_deltas(train_features, train_labels, selected_epochs, num_clients):
    bias_deltas = {rnd: {client_idx: [] for client_idx in range(num_clients)} for rnd in selected_epochs}

    for rnd in selected_epochs:
        print(f"Processing round {rnd}")
        epoch_dir = os.path.join(model_dir, f"round_{rnd}")

        for client_idx in range(num_clients):
            model_path = os.path.join(epoch_dir, f"client_{client_idx}.pth")
            model = create_model().to(device)
            model.load_state_dict(torch.load(model_path))

            for record_idx in range(len(train_features)):
                record_features = torch.tensor([train_features[record_idx]], dtype=torch.float32).to(device)
                record_label = torch.tensor([train_labels[record_idx]], dtype=torch.float32).to(device)

                bias_delta = compute_bias_delta(model, record_features, record_label)
                bias_deltas[rnd][client_idx].append(bias_delta)

    return bias_deltas

# Main execution
bias_deltas = construct_bias_deltas(train_features, train_labels, selected_epochs, num_clients)

# Save bias_deltas to a JSON file
with open("bias_deltas.json", "w") as f:
    json.dump({rnd: {client: [delta.tolist() for delta in deltas]
                     for client, deltas in clients.items()}
               for rnd, clients in bias_deltas.items()}, f)

print("Bias deltas saved to 'bias_deltas.json'.")
