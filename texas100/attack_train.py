# train_attack_model.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import logging 
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the prepared attack data
data = np.load("prepared_attack_data.npz")
X_attack = data['X_attack']
y_attack = data['y_attack']

# Split attack dataset
X_train, X_val, y_train, y_val = train_test_split(X_attack, y_attack, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# DataLoader setup
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)

# Define the attack model (CNN + FCN structure)
class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.flatten_size = 128 * 100
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
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
