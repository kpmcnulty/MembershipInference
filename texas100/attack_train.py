import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import logging
import matplotlib.pyplot as plt
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
print(len(member_vectors))
print(len(non_member_vectors))
amplification_factors = [1, 2, 5]
# def amplify_bias_deltas(bias_deltas, amplification_factor):
#     amplified_deltas = []
#     for delta in bias_deltas:
#         amplified_delta = np.exp(amplification_factor * delta) - 1
#         amplified_deltas.append(amplified_delta)
#     return amplified_deltas

# # Create a figure to plot bias changes for different amplification factors
# plt.figure(figsize=(15, len(amplification_factors) * 4))

        
# for idx, factor in enumerate(amplification_factors):
#     # Amplify the bias changes using the given amplification factor
#     amplified_deltas = amplify_bias_deltas(member_vectors[:1000].flatten(), factor)
#     #amplified_deltas_array = np.concatenate(amplified_deltas)  # Flatten to get a 1D array for distribution

#     # Plot the histogram for the current amplification factor
#     plt.subplot(len(amplification_factors), 1, idx + 1)
#     plt.hist(amplified_deltas, bins=30, alpha=0.7, color='b')
#     plt.title(f"Histogram of Bias Changes (Amplification Factor = {factor})")
#     plt.xlabel("Bias Change Value")
#     plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()
# Split the attack dataset into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(attack_vectors, attack_labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  # Use float type for BCELoss
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)  # Use float type for BCELoss
print(len(X_train_tensor[0]))
# DataLoader setup
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)

# Define the attack model (CNN + FCN structure)
class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        
        # Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        #self.fc1 = nn.Linear(64 * 100, 128)  # Adjusted for input size 800
        self.fc1 = nn.Linear(64 * 112, 128)  # Adjust input size after convolution and pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Single output node for binary classification
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary output
    
    def forward(self, x):
        # Convolutional layers with ReLU and MaxPooling
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        # Flatten the output of convolutional layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU activation
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # Output layer with sigmoid activation
        x = self.sigmoid(self.fc3(x))
        
        return x

# Initialize and train the attack model
attack_model = AttackModel().to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(attack_model.parameters(), lr=0.0005)

# Training loop for the attack model
epochs = 500
best_accuracy = 0
for epoch in range(epochs):
    logger.info(f"Epoch {epoch+1}/{epochs}")
    attack_model.train()
    for X_batch, y_batch in train_loader:
        # Add channel dimension to X_batch to match Conv1d input requirements
        X_batch, y_batch = X_batch.unsqueeze(1).to(device), y_batch.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = attack_model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Validation
    attack_model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.unsqueeze(1).to(device)  # Add channel dimension for validation as well
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
