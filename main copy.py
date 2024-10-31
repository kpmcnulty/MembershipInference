import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from torchvision import datasets, transforms
from flwr.common import Context
from logging import INFO, DEBUG
from flwr.common.logger import log


fl.common.logger.configure(identifier="federated", filename="log.txt")
# Load dataset
dataset = np.load("./texas100.npz")
features = dataset['features']
labels = dataset['labels']  # onehot encoded

bias_changes = {}
model = torch.nn.Sequential(
    torch.nn.Linear(6169, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 100, bias=True),
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        log(INFO, f"Initialized client {self.client_id}")
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
class MaliciousFlowerClient(FlowerClient):
    def __init__(self, model, client_id):
        super().__init__(model, client_id)
        self.previous_bias = None
        self.bias_deltas = []
        self.m = 2  # Amplification param
        log(INFO, f"Malicious client {self.client_id}: starting")
    def extract_bias(self):
        # Explicitly extract the bias from the last layer
        for name, param in self.model.named_parameters():
            if 'bias' in name:
                return param.detach().cpu().numpy()
        return None

    def calculate_bias_change(self, current_bias):
        if self.previous_bias is None:
            self.previous_bias = current_bias
            return np.zeros_like(current_bias)
        bias_change = current_bias - self.previous_bias
        self.previous_bias = current_bias
        return bias_change
    
    def amplify_bias_change(self, bias_change):
        return np.exp(self.m * bias_change) - 1

    def fit(self, parameters, config):
        bias_changes_matrix = []

        self.set_parameters(parameters)
        self.model.train()

        # Print initial bias

        # Train as usual
        for epoch in range(100):
            initial_bias = self.extract_bias()  # Capture the initial bias at the start of the epoch

            log(INFO, f"Initial bias: {initial_bias}")
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                X, y = batch
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

            # Capture bias after training the epoch and calculate bias change
            final_bias = self.extract_bias()
            log(INFO,"Final bias:")
            log(INFO,final_bias)
            delta_bias = final_bias - initial_bias
            log(INFO, "delta:")
            log(INFO, delta_bias)
            log(INFO, "delta_bias_list:")
            log(INFO, delta_bias.tolist())
            bias_changes_matrix.append(delta_bias.tolist())  # Append bias change for each epoch
        
        # Step 2: Construct a temporal matrix of Δbias differences between consecutive epochs
        delta_bias_temporal = np.diff(np.array(bias_changes_matrix), axis=0)  # Equation (7)
        log(INFO, delta_bias_temporal)
        #amplification function
        amplified_delta_bias = np.exp(self.m * delta_bias_temporal) - 1  # Equation (9)
        log(INFO, amplified_delta_bias)
        # Save amplified bias differences to a file for attack analysis
        output_path = "amplified_bias_changes.json"

        
        formatted_bias = [[f"{element:.6f}" for element in row] for row in amplified_delta_bias]
        log(INFO,formatted_bias[0])
        with open(output_path, "w") as f:

            json.dump(formatted_bias, f)

        log(INFO, f"Amplified bias changes saved to {output_path}")
        return self.get_parameters(), len(self.dataset), {}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        # Evaluate as usual
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in self.dataloader:
                outputs = self.model(X)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(y, 1)
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
    if cid == "4":  # Malicious client ID
        return MaliciousFlowerClient(model, int(cid)).to_client()
    else:
        return FlowerClient(model, int(cid)).to_client()
# Start Flower Simulation
fl.simulation.start_simulation(
    ray_init_args={"num_cpus": 8, "num_gpus": 0},
    client_fn=client_fn,
    num_clients=5,
    client_resources={"num_cpus": 1,"num_gpus": 0},
    config=fl.server.ServerConfig(num_rounds=10)
)
torch.save(model.state_dict(), "final_model_parameters.pth")
print(bias_changes)