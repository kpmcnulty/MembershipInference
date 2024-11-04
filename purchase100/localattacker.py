import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='attack.log', encoding='utf-8', level=logging.INFO)
# Load bias log and attack dataset splits
with open("client_4_bias_log.json", "r") as f:
    bias_log = json.load(f)

with open("attack_splits.json", "r") as f:
    attack_splits = json.load(f)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Parameters for attack vector construction
amplification_factor = 5  # Based on the paper's recommendation
interval = 5  # Every 5 epochs
logger.info(len(attack_splits))
# Step 1: Select epochs at specified intervals and extract last layer biases
selected_epochs = list(range(0, len(bias_log), interval))
last_layer_biases = [np.array(bias_log[epoch][-1]) for epoch in selected_epochs]  # Each entry is for all data points

# Step 2: Compute bias changes (deltas) for each data point across selected epochs
bias_deltas = []
for i in range(1, len(selected_epochs)):
    delta = last_layer_biases[i] - last_layer_biases[i - 1]  # Shape: (num_data_points, last_layer_bias_dim)
    bias_deltas.append(delta)
# Step 2: Amplify features using the exponential function
# Reshape each delta to be 2-dimensional (one row per data point)
amplified_deltas = [np.exp(amplification_factor * delta).reshape(-1, 1) - 1 for delta in bias_deltas]

# Step 3: Concatenate amplified deltas along the second dimension to form attack vectors per data point
attack_vectors = np.concatenate(amplified_deltas, axis=1)  # Final shape: (num_data_points, bias_dim * num_intervals)


# Prepare attack dataset based on the splits in attacker
train_mem_indices = attack_splits["attack_train_mem_indices"]
train_non_indices = attack_splits["attack_train_non_indices"]
test_mem_indices = attack_splits["attack_test_mem_indices"]
test_non_indices = attack_splits["attack_test_non_indices"]

# Construct training and testing data
X_train = np.concatenate([attack_vectors[train_mem_indices], attack_vectors[train_non_indices]])
y_train = np.array([1] * len(train_mem_indices) + [0] * len(train_non_indices))
X_test = np.concatenate([attack_vectors[test_mem_indices], attack_vectors[test_non_indices]])
y_test = np.array([1] * len(test_mem_indices) + [0] * len(test_non_indices))

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) 
y_train = torch.tensor(y_train, dtype=torch.long)  
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define CNN-based attack model based on description in paper
class AttackModel(nn.Module):
    def __init__(self, input_dim):
        super(AttackModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * ((input_dim - 2) // 2 - 2), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.softmax(x)

# Instantiate attack model
input_dim = X_train.shape[2]
attack_model = AttackModel(input_dim=input_dim).to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

# Training loop for the attack model
for epoch in range(20): #can adjust
    attack_model.train()
    optimizer.zero_grad()
    outputs = attack_model(X_train.to(device))
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Evaluate the attack model
attack_model.eval()
with torch.no_grad():
    outputs = attack_model(X_test.to(device))
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test.to(device)).float().mean()
    print(f"Attack Model Test Accuracy: {accuracy.item() * 100:.2f}%")
