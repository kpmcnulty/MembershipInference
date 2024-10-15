import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from torchvision import datasets, transforms

# Load dataset
dataset = np.load("./texas100.npz")
features = dataset['features']
labels = dataset['labels']  # onehot encoded
num_participants = 5

local_datasets = []
for i in range(num_participants):
    start_index = int(i * (features.shape[0] / num_participants))
    end_index = int((i + 1) * (features.shape[0] / num_participants))
    local_datasets.append((features[start_index:end_index], labels[start_index:end_index]))

# Define the model based on the 5.2 target model
model = torch.nn.Sequential(
    torch.nn.Linear(67330, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 100),
    torch.nn.Softmax(dim=1)
)

# Define Flower client using PyTorch
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_data):
        self.model = model
        self.train_data = train_data
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, val in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(val).float()

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        X_train, y_train = self.train_data
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        # Training loop
        self.optimizer.zero_grad()
        outputs = self.model(X_train)
        loss = self.criterion(outputs, y_train.argmax(dim=1))  # Convert one-hot to class labels
        loss.backward()
        self.optimizer.step()

        return self.get_parameters(), len(X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        X_test, y_test = self.train_data
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(X_test)
            loss = self.criterion(outputs, y_test.argmax(dim=1))  # Convert one-hot to class labels
            accuracy = (outputs.argmax(dim=1) == y_test.argmax(dim=1)).float().mean().item()

        return float(loss), len(X_test), {"accuracy": accuracy}

# Start Flower client
def start_federated_learning():
    clients = []
    for i in range(num_participants):
        train_data = local_datasets[i]
        clients.append(FlowerClient(model, train_data))

    # Start Flower client
    fl.client.start_client(
        server_address="localhost:8080",
        client=clients[0]  # For now, we only start one client for testing
    )

# Entry point
if __name__ == "__main__":
    start_federated_learning()

