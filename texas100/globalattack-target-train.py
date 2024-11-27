import os
import torch
import numpy as np
import flwr as fl
from torch import nn
from torch.utils.data import DataLoader
import io
# Ensure the necessary imports and datasets are set up
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Dataset and Model Preparation
dataset = np.load("./texas100.npz")
features, global_labels = dataset['features'], dataset['labels']

indices = np.random.permutation(len(features))
train_indices = indices[:30000]
test_indices = indices[30000:60000]
train_features, train_labels = features[train_indices], global_labels[train_indices]
test_features, test_labels = features[test_indices], global_labels[test_indices]

num_clients = 5
client_data_indices = np.array_split(np.arange(len(train_features)), num_clients)

np.save("global_train_indices.npy", train_indices)

# Map each data record to the client (participant) that owns it
ground_truth_sources = np.zeros(len(train_features), dtype=int)

for client_id, indices in enumerate(client_data_indices):
    ground_truth_sources[indices] = client_id
print(ground_truth_sources.shape)
# Save ground truth sources to a file for validation
np.save("ground_truth_sources.npy", ground_truth_sources)

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
    indices = client_data_indices[cid]
    dataset = Texas100Dataset(train_features[indices], train_labels[indices])
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return FlowerClient(create_model(), cid, data_loader)

class MaliciousAggregation(fl.server.strategy.FedAvg):
    def __init__(self, selected_epochs, model_dir="models", **kwargs):
        super().__init__(**kwargs)
        self.selected_epochs = selected_epochs
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def aggregate_fit(self, rnd: int, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)

        # Save models at selected epochs
        if rnd in self.selected_epochs:
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

# Start simulation
selected_rounds = [1, 2, 3, 4, 5, 6, 7, 8, 9,10]
strategy = MaliciousAggregation(selected_epochs=selected_rounds)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=10),
    ray_init_args={"num_cpus": 8, "num_gpus": 1},
)
