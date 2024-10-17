import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from torchvision import datasets, transforms
from flwr.common import Context
# Load dataset
dataset = np.load("./texas100.npz")
features = dataset['features']
labels = dataset['labels']  # onehot encoded

model = torch.nn.Sequential(
    torch.nn.Linear(6169, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 100),
    torch.nn.Softmax(dim=1)
)
class Texas100Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


subset_percentage = 0.7  # 20% of the data
num_samples = int(len(features) * subset_percentage)
subset_features = features[:num_samples]
subset_labels = labels[:num_samples]
client_data_splits = np.array_split(range(len(subset_features)), 5)

# Define Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, client_id):
        self.model = model
        self.client_id = client_id
        self.dataset = Texas100Dataset(subset_features[client_data_splits[client_id]], subset_labels[client_data_splits[client_id]])
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=64, shuffle=True,num_workers=0)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]
    
    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=torch.float32)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(100):  # Train for 100 epochs
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                X, y = batch
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
                outputs = self.model(X)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(y, 1)  # Convert one-hot to labels
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total, len(self.dataset), {}

# Initialize and start clients
def client_fn(cid: str):
    model = torch.nn.Sequential(
        torch.nn.Linear(6169, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 100),
        torch.nn.Softmax(dim=1)
    )
    return FlowerClient(model, int(cid)).to_client()

# Start Flower Simulation
fl.simulation.start_simulation(
    ray_init_args={"num_cpus": 8, "num_gpus": 0},
    client_fn=client_fn,
    num_clients=5,
    client_resources={"num_cpus": 1,"num_gpus": 0},
    config=fl.server.ServerConfig(num_rounds=10)
)