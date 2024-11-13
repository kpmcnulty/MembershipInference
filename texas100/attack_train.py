# train_attack_model.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import logging 

import torch.nn.functional as F
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the prepared attack data
attack_vectors = np.load("attack_vectors.npy")
attack_labels = np.load("attack_labels.npy")
# Inspect the data to see differences between members and non-members
member_vectors = attack_vectors[attack_labels == 1]
non_member_vectors = attack_vectors[attack_labels == 0]

# Calculate some basic statistics to compare members and non-members
mean_member_vector = np.mean(member_vectors, axis=0)
mean_non_member_vector = np.mean(non_member_vectors, axis=0)

std_member_vector = np.std(member_vectors, axis=0)
std_non_member_vector = np.std(non_member_vectors, axis=0)

print(attack_vectors.shape)
# Split the attack dataset into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(attack_vectors, attack_labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
logger.info(X_train_tensor.shape)
logger.info(y_train_tensor.shape)

# DataLoader setup
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)

# Define the attack model (CNN + FCN structure)
class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        # Adjust the input size of the first fully connected layer to match the attack vector size (900)
        self.fc1 = nn.Linear(900, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Use sigmoid for binary classification
        return x

# Initialize and train the attack model
attack_model = AttackModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(attack_model.parameters(), lr=0.0005)

# Training loop for the attack model
epochs = 500
best_accuracy = 0
for epoch in range(epochs):
    logger.info(f"Epoch {epoch+1}/{epochs}")
    attack_model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = attack_model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    attack_model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = attack_model(X_batch).squeeze()
            predictions = (outputs > 0.5).cpu().numpy()
            y_pred.extend(predictions)
            y_true.extend(y_batch.numpy())

    # Accuracy calculation
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"Validation Accuracy: {accuracy:.4f}")

    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(attack_model.state_dict(), "best_attack_model.pth")
        logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
