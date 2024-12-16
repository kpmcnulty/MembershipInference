import os
import json
import torch
import numpy as np
from torch import nn
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
num_epochs = 5 #########
# Configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_dir = f"models{num_epochs}"
selected_epochs = [5, 10, 20, 25, 30, 35, 45, 50, 60, 85]
num_clients = 5

# Load dataset and splits
dataset = np.load("../texas100.npz")
with open(f"../dataset_splits{num_epochs}.json", "r") as f:
    dataset_splits = json.load(f)
    client_splits = dataset_splits["client_splits"]

# Get all training indices in order
train_indices = []
client_labels = []  # Store labels while we build indices
for client_id, indices in client_splits.items():
    client_num = int(client_id.split('_')[1])
    train_indices.extend(indices)
    client_labels.extend([client_num] * len(indices))

# Load features for training data
train_features = dataset["features"][train_indices]
train_labels = dataset["labels"][train_indices]
labels = np.array(client_labels)  # These are our client labels

logger.info(f"Loaded dataset with {len(train_features)} samples")

def create_model():
    return nn.Sequential(
        nn.Linear(6169, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 100),
        nn.Softmax(dim=1),
    )

def compute_bias_delta_batch(model, features_batch, labels_batch):
    """Compute bias changes through simulated training for a batch of records"""
    bias_before = model[-2].bias.detach().cpu().clone().numpy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for _ in range(5):  # Simulate training for 5 epochs
        optimizer.zero_grad()
        outputs = model(features_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

    bias_after = model[-2].bias.detach().cpu().numpy()
    return bias_after - bias_before
def construct_attack_vectors(train_features, train_labels, selected_epochs, num_clients):
    num_samples = len(train_features)
    num_epochs = len(selected_epochs)
    bias_dim = 100

    # Initialize attack vectors array
    attack_vectors = np.zeros((num_samples, num_clients, num_epochs, bias_dim))

    logger.info("Preloading models for all clients and epochs...")
    # Preload models
    preloaded_models = {
        epoch: {
            client_idx: torch.load(os.path.join(model_dir, f"round_{epoch}", f"client_{client_idx}.pth"))
            for client_idx in range(num_clients)
        }
        for epoch in selected_epochs
    }

    logger.info(f"Preloaded models for {len(selected_epochs)} epochs and {num_clients} clients.")

    for epoch_idx, epoch in enumerate(selected_epochs):
        for client_idx in range(num_clients):
            model = create_model().to(device)
            model.load_state_dict(preloaded_models[epoch][client_idx])
            model.train()

            logger.info(f"Processing epoch {epoch}, client {client_idx}...")
            for sample_idx in range(num_samples):
                # Prepare data for the single sample
                feature_sample = torch.tensor(
                    train_features[sample_idx:sample_idx + 1], dtype=torch.float32
                ).to(device)

                label_sample = torch.tensor(
                    [np.argmax(train_labels[sample_idx])], dtype=torch.long
                ).to(device)

                # Compute bias delta for the single sample
                bias_delta = compute_bias_delta_batch(model, feature_sample, label_sample)

                # Store in the attack vectors array
                attack_vectors[sample_idx, client_idx, epoch_idx, :] = bias_delta

                # Log progress
    logger.info("Attack vector construction completed.")
    return attack_vectors

def plot_feature_distributions(attack_vectors, labels, save_path=f"global_attack_distributions{num_epochs}/"):
    os.makedirs(save_path, exist_ok=True)
    
    # Overall magnitude distribution per client
    magnitudes = np.mean(np.abs(attack_vectors), axis=(1, 2))
    
    plt.figure(figsize=(12, 6))
    for i in range(num_clients):
        client_magnitudes = magnitudes[labels == i]
        sns.histplot(data=client_magnitudes, 
                    label=f'Client {i}',
                    common_norm=True, 
                    stat='density',
                    alpha=0.6)
    
    plt.title('Distribution of Global Attack Vector Magnitudes by Client')
    plt.xlabel('Mean Magnitude')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, f'global_attack_magnitude_distribution{num_epochs}.png'))
    plt.close()
    
    # Per-epoch distributions
    for epoch_idx, epoch in enumerate(selected_epochs):
        epoch_vectors = attack_vectors[:, epoch_idx, :]
        epoch_magnitudes = np.mean(np.abs(epoch_vectors), axis=1)
        
        plt.figure(figsize=(12, 6))
        for i in range(num_clients):
            client_magnitudes = epoch_magnitudes[labels == i]
            sns.histplot(data=client_magnitudes,
                        label=f'Client {i}',
                        common_norm=True,
                        stat='density',
                        alpha=0.6)
        
        plt.title(f'Distribution of Attack Vector Magnitudes - Epoch {epoch}')
        plt.xlabel('Mean Magnitude')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, f'magnitude_distribution{num_epochs}_epoch_{epoch}.png'))
        plt.close()

    # Log statistics per client
    logger.info("\nAttack Vector Statistics:")
    logger.info(f"Overall shape: {attack_vectors.shape}")
    
    for i in range(num_clients):
        client_magnitudes = magnitudes[labels == i]
        client_stats = {
            'mean': np.mean(client_magnitudes),
            'std': np.std(client_magnitudes),
            'min': np.min(client_magnitudes),
            'max': np.max(client_magnitudes)
        }
        
        logger.info(f"\nClient {i} Statistics:")
        for key, value in client_stats.items():
            logger.info(f"{key}: {value:.4f}")

# Main execution
logger.info("Starting attack vector construction...")
attack_vectors = construct_attack_vectors(train_features, train_labels, selected_epochs, num_clients)

# Save results - attack_vectors[i] and labels[i] will correspond to the same sample
np.save(f"global_attack_vectors{num_epochs}.npy", attack_vectors)
np.save(f"global_attack_labels{num_epochs}.npy", labels)

logger.info("Attack vectors saved to 'global_attack_vectors{num_epochs}.npy'")
logger.info("Attack labels saved to 'global_attack_labels{num_epochs}.npy'")
