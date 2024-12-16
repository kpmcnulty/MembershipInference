import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import matplotlib.pyplot as plt
num_epochs = 5 #########
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load the prepared attack data
attack_vectors = np.load(f"amplified_vectors{num_epochs}.npy")
attack_labels = np.load(f"attack_labels{num_epochs}.npy")

print(f"Original input shape: {attack_vectors.shape}")

# Instead of using the indices from dataset_splits, we'll split based on the order
# in which attack_vectors were created (first 30000 are members, last 30000 are non-members)
member_range = np.arange(30000)  # First 30000 are members
non_member_range = np.arange(30000, 60000)  # Last 30000 are non-members

# Use first 24000 members and 24000 non-members for training
train_member_indices = member_range[:24000]
test_member_indices = member_range[24000:]
train_non_member_indices = non_member_range[:24000]
test_non_member_indices = non_member_range[24000:]


# Combine training and testing indices
train_indices = np.concatenate([train_member_indices, train_non_member_indices])
test_indices = np.concatenate([test_member_indices, test_non_member_indices])

# Split the data
X_train = attack_vectors[train_indices]
y_train = attack_labels[train_indices]
X_val = attack_vectors[test_indices]
y_val = attack_labels[test_indices]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# DataLoader setup
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)

logger.info(f"Training set size: {len(X_train)} ({np.sum(y_train == 1)} members, {np.sum(y_train == 0)} non-members)")
logger.info(f"Test set size: {len(X_val)} ({np.sum(y_val == 1)} members, {np.sum(y_val == 0)} non-members)")

# Define the attack model (simpler architecture with dropout)
class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        
        # Multiple conv-relu-pool modules as mentioned in the paper
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # Linear-relu modules
        self.fc1 = nn.Linear(3600,32)  
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 2)  
        
        # Binary classification output
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # Conv-relu-pool modules
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Linear-relu modules
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Final classification probability
        x = self.softmax(self.fc3(x))
        
        return x

# Update training loop for binary classification
def train_attack_model(attack_model, train_loader, val_loader, epochs=500):
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
    
    best_accuracy = 0.0
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Initial accuracy
    attack_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.unsqueeze(1).to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)
            outputs = attack_model(X_batch)
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    initial_accuracy = correct / total
    logger.info(f"Initial accuracy (epoch 0): {initial_accuracy:.4f}")
    
    for epoch in range(epochs):
        attack_model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.unsqueeze(1).to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = attack_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        attack_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.unsqueeze(1).to(device)
                y_batch = y_batch.float().unsqueeze(1).to(device)
                outputs = attack_model(X_batch)
                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)
        
        logger.info(f"Epoch {epoch+1}/{epochs}")
        logger.info(f"Training Loss: {avg_train_loss:.4f}")
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(attack_model.state_dict(), f"best_attack_model{num_epochs}.pth")
            logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
    
    return train_losses, train_accuracies, val_accuracies

# Initialize and train the attack model
attack_model = AttackModel().to(device)
train_losses, train_accuracies, val_accuracies = train_attack_model(attack_model, train_loader, val_loader)

# Plot training curves
plt.figure(figsize=(12, 4))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Time')
plt.legend()

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
plt.plot(range(len(val_accuracies)), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Time')
plt.legend()

plt.tight_layout()
plt.savefig(f'training_metrics{num_epochs}.png')
plt.close()
