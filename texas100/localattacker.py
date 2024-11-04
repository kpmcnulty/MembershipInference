import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename='attack.log', encoding='utf-8', level=logging.INFO)

# Load bias log and attack dataset splits
with open("client_4_bias_log.json", "r") as f:
    bias_log = json.load(f)

with open("attack_splits.json", "r") as f:
    attack_splits = json.load(f)
# Load updated bias log with thousands of data points
with open("client_4_bias_log.json", "r") as f:
    bias_log = json.load(f)

selected_intervals = range(5, 100, 5)
attack_vectors = []

# Construct attack vectors for each data point
for data_point_idx in bias_log:
    data_point_vectors = []
    previous_bias = None
    
    for epoch_idx in selected_intervals:
        # Get last layer bias for the current epoch
        last_layer_bias = np.array(bias_log[data_point_idx][epoch_idx // 5])  # Assuming epochs are every 5 steps
        
        if previous_bias is not None:
            # Calculate Δbias between epochs
            delta_bias = last_layer_bias - previous_bias
            data_point_vectors.append(delta_bias)  # Append Δbias for this data point

        previous_bias = last_layer_bias
    
    # Concatenate Δbias vectors to form attack vector for this data point
    attack_vector = np.concatenate(data_point_vectors)
    attack_vectors.append(attack_vector)

attack_vectors = np.array(attack_vectors)

# Ensure we have thousands of attack vectors matching the `attack_splits` indices
train_mem_indices = attack_splits["attack_train_mem_indices"]
train_non_indices = attack_splits["attack_train_non_indices"]
test_mem_indices = attack_splits["attack_test_mem_indices"]
test_non_indices = attack_splits["attack_test_non_indices"]

X_train = np.concatenate([attack_vectors[train_mem_indices], attack_vectors[train_non_indices]])
y_train = np.array([1] * len(train_mem_indices) + [0] * len(train_non_indices))
X_test = np.concatenate([attack_vectors[test_mem_indices], attack_vectors[test_non_indices]])
y_test = np.array([1] * len(test_mem_indices) + [0] * len(test_non_indices))

# Continue with model training as previously outlined...

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
for epoch in range(20):  # Can adjust epochs
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
