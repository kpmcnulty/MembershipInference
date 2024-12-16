from logging import log
import os
import torch
import numpy as np
import flwr as fl
from torch import nn
from torch.utils.data import DataLoader
import io
import json
import matplotlib.pyplot as plt

# Ensure the necessary imports and datasets are set up
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load dataset
dataset = np.load("./texas100.npz")
features = dataset['features']
global_labels = dataset['labels']

# Define dataset splits
total_size = len(features)
train_size = 30000
test_size = 30000

# Shuffle and split the dataset
indices = np.random.permutation(total_size)

# Define target model indices
train_indices = indices[:train_size]
test_indices = indices[train_size:train_size + test_size]

# Define non-member indices
non_member_indices = np.setdiff1d(indices, train_indices)

# Combine members and non-members for attack model
member_indices = train_indices.tolist()
non_member_indices = non_member_indices.tolist()

# Create client splits
num_clients = 5
client_indices = np.array_split(train_indices, num_clients)
client_splits = {
    f"client_{i}": client_indices[i].tolist() 
    for i in range(num_clients)
}

# Create combined splits dictionary
dataset_splits = {
    "attack_splits": {
        "member_indices": member_indices,
        "non_member_indices": non_member_indices,
    },
    "client_splits": client_splits
}

# Save all splits to a single JSON file
with open("dataset_splits.json", "w") as f:
    json.dump(dataset_splits, f)


def create_model():
    return nn.Sequential(
        nn.Linear(6169, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 100),
        nn.Softmax(dim=1)
    )

class Texas100Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, client_id, data_loader):
        self.model = model.to(device)
        self.client_id = client_id
        self.data_loader = data_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        print(f"Initialized client {self.client_id} with {len(self.data_loader.dataset)} samples")


    def get_parameters(self, config=None):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=torch.float32).to(device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(14): #from global attack highest local epoch accuracy  
            for X, y in self.data_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X)
                labels = torch.argmax(y, dim=1) if y.dim() > 1 else y
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(self.data_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in self.data_loader:
                outputs = self.model(X)
                _, predicted = torch.max(outputs, 1)
                labels = torch.argmax(y, dim=1) if y.dim() > 1 else y
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total, len(self.data_loader.dataset), {}

def client_fn(cid: str):
    cid = int(cid)
    indices = client_splits[f"client_{cid}"]
    client_features = features[indices]
    client_labels = global_labels[indices]
    dataset = Texas100Dataset(client_features, client_labels)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return FlowerClient(create_model(), cid, data_loader)

class MaliciousAggregation(fl.server.strategy.FedAvg):
    def __init__(self, model_dir="models", **kwargs):
        super().__init__(**kwargs)
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def aggregate_fit(self, rnd: int, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)

        # Save models at selected epochs
   
        self.save_local_models(rnd, results)

        return aggregated_weights

    def save_local_models(self, rnd, results):
        epoch_dir = os.path.join(self.model_dir, f"round_{rnd}")
        os.makedirs(epoch_dir, exist_ok=True)

        for client_idx, (_, fit_res) in enumerate(results):
            model_params = [
                torch.tensor(np.load(io.BytesIO(tensor))).to(device)
                for tensor in fit_res.parameters.tensors
            ]
            model = create_model().to(device)
            self.set_model_params(model, model_params)
            torch.save(model.state_dict(), os.path.join(epoch_dir, f"client_{client_idx}.pth"))

    @staticmethod
    def set_model_params(model, params_list):
        with torch.no_grad():
            for param, new_param in zip(model.parameters(), params_list):
                param.data = new_param.data.clone()
        return model

# Add these before the strategy definition
accuracy_per_round = {'train': [], 'test': []}
loss_per_round = {'train': [], 'test': []}

def evaluate_fn(server_round: int, parameters, config):
    model = create_model().to(device)
    for param, new_param in zip(model.parameters(), parameters):
        param.data = torch.tensor(new_param, dtype=torch.float32).to(device)

    # Create both train and test datasets
    train_dataset = Texas100Dataset(features[train_indices], global_labels[train_indices])
    test_dataset = Texas100Dataset(features[test_indices], global_labels[test_indices])
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    def evaluate_dataset(dataloader, split_name):
        model.eval()
        correct, total = 0, 0
        total_loss = 0.0
        with torch.no_grad():
            for X, y in dataloader:
                outputs = model(X)
                loss = nn.CrossEntropyLoss()(outputs, y.argmax(dim=1))
                total_loss += loss.item() * X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(y, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        average_loss = total_loss / total
        
        accuracy_per_round[split_name].append(accuracy)
        loss_per_round[split_name].append(average_loss)
        
        return accuracy, average_loss

    train_acc, train_loss = evaluate_dataset(train_dataloader, 'train')
    test_acc, test_loss = evaluate_dataset(test_dataloader, 'test')
    
    print(f"Round {server_round}")
    print(f"Train - Accuracy: {train_acc:.4f}, Loss: {train_loss:.4f}")
    print(f"Test  - Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

    return test_loss, {"accuracy": test_acc}

def plot_metrics():
    rounds = range(1, len(accuracy_per_round['train']) + 1)

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracy_per_round['train'], label="Train Accuracy", linestyle='--')
    plt.plot(rounds, accuracy_per_round['test'], label="Test Accuracy")
    plt.title("Model Accuracy Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.savefig('global_accuracy_plot.png')
    plt.close()

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, loss_per_round['train'], label="Train Loss", linestyle='--')
    plt.plot(rounds, loss_per_round['test'], label="Test Loss")
    plt.title("Model Loss Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig('global_loss_plot.png')
    plt.close()

# Update strategy to use evaluate_fn
strategy = MaliciousAggregation(
    evaluate_fn=evaluate_fn
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=100),
    ray_init_args={"num_cpus": 8, "num_gpus": 1},
)

# Add this after the simulation
plot_metrics()
