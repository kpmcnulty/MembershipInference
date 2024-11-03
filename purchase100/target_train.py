import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from logging import INFO, DEBUG
from flwr.common.logger import log
from flwr.server.strategy import FedAvg

# Check for MPS availability ### for group: this is a mac m chip thing, if you have a gpu you can use CUDA or something, if not, ignore this and use cpu only
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
log(INFO, f"Using device: {device}")

fl.common.logger.configure(identifier="federated", filename="log.txt")

# Load dataset
dataset = np.load("./purchase100.npz")
features = dataset['features']
global_labels = dataset['labels']  

# Define dataset splits according to Table 12 for Purchase100
target_train_size = 10000
target_test_size = 10000
attack_train_mem_size = 8000
attack_train_non_size = 8000
attack_test_mem_size = 2000
attack_test_non_size = 2000

# Shuffle and split the dataset
indices = np.random.permutation(len(features))
target_train_indices = indices[:target_train_size]
target_test_indices = indices[target_train_size:target_train_size + target_test_size]
attack_train_mem_indices = indices[target_train_size + target_test_size:target_train_size + target_test_size + attack_train_mem_size]
attack_train_non_indices = indices[target_train_size + target_test_size + attack_train_mem_size:target_train_size + target_test_size + attack_train_mem_size + attack_train_non_size]
attack_test_mem_indices = indices[target_train_size + target_test_size + attack_train_mem_size + attack_train_non_size:target_train_size + target_test_size + attack_train_mem_size + attack_train_non_size + attack_test_mem_size]
attack_test_non_indices = indices[target_train_size + target_test_size + attack_train_mem_size + attack_train_non_size + attack_test_mem_size:]

# Save attack dataset splits for use in local attack training
attack_splits = {
    "attack_train_mem_indices": attack_train_mem_indices.tolist(),
    "attack_train_non_indices": attack_train_non_indices.tolist(),
    "attack_test_mem_indices": attack_test_mem_indices.tolist(),
    "attack_test_non_indices": attack_test_non_indices.tolist()
}
with open("attack_splits.json", "w") as f:
    json.dump(attack_splits, f)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_model(): #model from paper
    return torch.nn.Sequential(
        torch.nn.Linear(600, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 100),
        torch.nn.Softmax(dim=1)
    )

# Define honest client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, client_id):
        self.model = model.to(device)
        self.client_id = client_id
        self.dataset = Dataset(features[target_train_indices], global_labels[target_train_indices])
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=True, num_workers=0)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        log(INFO, f"Initialized client {self.client_id}")
    
    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]
    
    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=torch.float32).to(device)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(100):  # Train for 100 epochs
            for batch in self.dataloader:
                X, y = batch
                X, y = X.to(device), y.to(device)  # IF you arent using mps or cuda delte .to(device)
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

# Define Malicious Flower Client with bias logging
class MaliciousFlowerClient(FlowerClient):
    def __init__(self, model, client_id):
        super().__init__(model, client_id)
        self.bias_log = [] 
        log(INFO, f"Initialized malicious client {self.client_id}")

    def extract_bias(self):
        return [param.detach().cpu().numpy().tolist() for name, param in self.model.named_parameters() if 'bias' in name]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        # Same as regular client but Log biases each epoch
        for epoch in range(100):
            for batch in self.dataloader:
                X, y = batch
                X, y = X.to(device), y.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

            self.bias_log.append(self.extract_bias())

        with open(f"client_{self.client_id}_bias_log.json", "w") as f:
            json.dump(self.bias_log, f)
        log(INFO, f"Biases logged for client {self.client_id}")
        
        return self.get_parameters(), len(self.dataset), {}

# Evaluation function for testing by server
def evaluate_fn(server_round: int, parameters, config):
    model = create_model().to(device)
    for param, new_param in zip(model.parameters(), parameters):
        param.data = torch.tensor(new_param, dtype=torch.float32).to(device)

    val_dataset = Dataset(features[target_test_indices], global_labels[target_test_indices])
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
    average_loss = total_loss / total  
    log(INFO, f"Round {server_round} - Centralized evaluation accuracy: {accuracy}, loss: {average_loss}")
    return average_loss, {"accuracy": accuracy}  #metrics

# initialize honest and malicious client
def client_fn(cid: str):
    model = create_model()
    if cid == "4":  # Make client 4 malicious for membership inference attack
        return MaliciousFlowerClient(model, int(cid)).to_client()
    else:
        return FlowerClient(model, int(cid)).to_client()
#eval strategy for sim
strategy = FedAvg(
    evaluate_fn=evaluate_fn
)

# Start Flower Simulation
fl.simulation.start_simulation(
    ray_init_args={"num_cpus": 8, "num_gpus": 1},
    client_fn=client_fn,
    num_clients=5,
    client_resources={"num_cpus": 1, "num_gpus": .2},
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy
)
