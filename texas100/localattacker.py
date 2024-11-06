import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import logging 
logger = logging.getLogger(__name__)
logging.basicConfig(filename='attack.log', encoding='utf-8', level=logging.INFO)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved attack splits, biases, and model snapshots
with open("attack_splits.json", "r") as f:
    attack_splits = json.load(f)

dataset = np.load("./texas100.npz")
features = dataset['features']
global_model_snapshots = torch.load("client_4_global_snapshots.pth")

# Define selected epochs for calculating bias deltas
selected_epochs = [5, 10, 20, 25, 30, 35, 45, 50, 60, 85]

def create_model(): #target model structure
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
logger.info(f"member-data{member_data.shape}")
logger.info(f"non_member_data{non_member_data.shape}")
# Calculate bias deltas
member_bias_deltas = calculate_bias_deltas(global_model_snapshots, member_data)
non_member_bias_deltas = calculate_bias_deltas(global_model_snapshots, non_member_data)
logger.info(f"member_bias_deltas{member_bias_deltas.shape}")
logger.info(f"non_member_bias_deltas{non_member_bias_deltas.shape}")
# Combine to form attack dataset
X_attack = np.concatenate([member_bias_deltas, non_member_bias_deltas], axis=0)
y_attack = np.concatenate([np.ones(member_bias_deltas.shape[0]), np.zeros(non_member_bias_deltas.shape[0])])
logger.info(f"X attack shape, {X_attack.shape}")
logger.info(f"y attack shape, {y_attack.shape}")
# Split attack dataset
X_train, X_val, y_train, y_val = train_test_split(X_attack, y_attack, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
logger.info(f"X train tensor shape, {X_train_tensor.shape}")
logger.info(f"y train tensor shape, {y_train_tensor.shape}")
logger.info(f"X val tensor shape, {X_val_tensor.shape}")
logger.info(f"y val tensor shape, {y_val_tensor.shape}")
# DataLoader setup
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64)
# Define the attack model (CNN + FCN structure)
class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        # Convolutional layers for 1D bias delta vectors
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # increased from 64 to 128

        # Flatten layer to prepare for fully connected layers
        self.flatten_size = 128 * 100  # 64 filters and 100 time steps after convolutions
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Adding a channel dimension for CNN input
        x = x.unsqueeze(1)  # Shape becomes (batch_size, 1, 100)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        
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
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = attack_model(X_batch).squeeze()
            predictions = (outputs > 0.5).cpu().numpy()  # Binary classification threshold
            y_pred.extend(predictions)
            y_true.extend(y_batch.numpy())
    
    # Calculate accuracy for the current epoch
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {accuracy:.4f}")

    # Save the model with the best validation accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(attack_model.state_dict(), "best_attack_model.pth")
        print(f"New best model saved with accuracy: {best_accuracy:.4f}") 
# Initialize attack model
attack_model = AttackModel(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(attack_model.parameters(), lr=0.0005)

# Training loop for the attack model
epochs = 500
best_accuracy = 0
for epoch in range(epochs):
    logger.info(epoch)
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
    logger.info(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {accuracy:.4f}")

    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(attack_model.state_dict(), "best_attack_model.pth")
        logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
