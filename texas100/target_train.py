import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from logging import INFO, DEBUG
from flwr.common.logger import log
from flwr.server.strategy import FedAvg
import copy

# Check for MPS availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
log(INFO, f"Using device: {device}")

fl.common.logger.configure(identifier="federated", filename="log.txt")

# Load dataset
dataset = np.load("./texas100.npz")
features = dataset['features']
global_labels = dataset['labels'] 
# Define dataset splits according to the Texas100 splits
target_train_size = 30000
target_test_size = 30000

# Shuffle and split the dataset
indices = np.random.permutation(60000)

# Step 1: Define target model indices
target_train_indices = indices[:target_train_size]
target_test_indices = indices[target_train_size:target_train_size + target_test_size]
print(len(target_train_indices))
print(len(target_test_indices))
print(np.intersect1d(target_train_indices, target_test_indices))

# Step 2: Define non-member indices
non_member_indices = np.setdiff1d(indices, target_train_indices)
# Step 3: Combine members and non-members for attack model
member_indices = target_train_indices.tolist()
non_member_indices = non_member_indices.tolist()
print(len(member_indices))
print(len(non_member_indices))
print(np.intersect1d(member_indices, non_member_indices))
attack_splits = {
    "member_indices": member_indices,
    "non_member_indices": non_member_indices,
}
with open("attack_splits.json", "w") as f:
    json.dump(attack_splits, f)

# Define the dataset
class Texas100Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_model():
    return torch.nn.Sequential(
        torch.nn.Linear(6169, 1024), 
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 100),
        torch.nn.Softmax(dim=1)
    )

# Define honest client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, client_id):
        self.model = model.to(device)
        self.client_id = client_id
        self.dataset = Texas100Dataset(features[target_train_indices], global_labels[target_train_indices])
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=True, num_workers=0)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        log(INFO, f"Initialized client {self.client_id}")
    
    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]
    
    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=torch.float32).to(device)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1): #more rounds, non-consequtive epochs
            for batch in self.dataloader:
                X, y = batch
                X, y = X.to(device), y.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.dataset), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in self.dataloader:
                X, y = X.to(device), y.to(device)
                outputs = self.model(X)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(y, 1)  # Convert one-hot to labels
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total, len(self.dataset), {}

class MaliciousFlowerClient(fl.client.NumPyClient):
    def __init__(self, model, client_id):
        self.model = model.to(device)
        self.client_id = client_id
        self.dataset = Texas100Dataset(features[target_train_indices], global_labels[target_train_indices])
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=True, num_workers=0)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        log(INFO, f"Initialized client {self.client_id}")
    
    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]
    
    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=torch.float32).to(device)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Save the model state before training

        pre_training_file = f"snaphot__{config['round']}.pth"
        torch.save(self.model.state_dict(), pre_training_file)
        log(INFO, f"Saved pre-training model state to {pre_training_file}")
        
        self.model.train()
        for epoch in range(1):  
            for batch in self.dataloader:
                X, y = batch
                X, y = X.to(device), y.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.dataset), {}
    
# Evaluation function for testing by server
def evaluate_fn(server_round: int, parameters, config):
    model = create_model().to(device)
    for param, new_param in zip(model.parameters(), parameters):
        param.data = torch.tensor(new_param, dtype=torch.float32).to(device)

    val_dataset = Texas100Dataset(features[target_test_indices], global_labels[target_test_indices])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    with torch.no_grad():
        for X, y in val_dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = nn.CrossEntropyLoss()(outputs, y.argmax(dim=1))  
            total_loss += loss.item() * X.size(0)  # batch loss
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(y, 1)  # Convert one-hot to labels
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    average_loss = total_loss / total  # Calculate average loss
    log(INFO, f"Round {server_round} - Centralized evaluation accuracy: {accuracy}, loss: {average_loss}")
    return average_loss, {"accuracy": accuracy}  # metrics

# initialize honest and malicious client
def client_fn(cid: str):
    model = create_model()
    if cid == "4":  # Make client 2 malicious for membership inference attack
        return MaliciousFlowerClient(model, int(cid)).to_client()
    else:
        return FlowerClient(model, int(cid)).to_client()
    
# eval stategy for sim
def fit_config(server_round: int):
    config = {
        "round": server_round
    }
    return config
strategy = FedAvg(
   on_fit_config_fn=fit_config,
   evaluate_fn=evaluate_fn
)

# Start Flower Simulation 
fl.simulation.start_simulation(
    ray_init_args={"num_cpus": 8, "num_gpus": 1},
    client_fn=client_fn,
    num_clients=5,
    client_resources={"num_cpus": 1, "num_gpus": .2},
    config=fl.server.ServerConfig(num_rounds=100), # 100 sharing 'epochs' so local training is non-consequtive 
    strategy=strategy
)
