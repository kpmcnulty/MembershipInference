import json
import numpy as np
import torch
import torch.nn as nn
import logging
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
num_epochs = 5 #########
# Load the saved splits and the dataset
with open(f"../dataset_splits{num_epochs}.json", "r") as f:
    dataset_splits = json.load(f)

dataset = np.load("../texas100.npz")
features = dataset['features']
labels = dataset['labels']

# Modify to use all entries
member_indices = np.array(dataset_splits["attack_splits"]["member_indices"])
non_member_indices = np.array(dataset_splits["attack_splits"]["non_member_indices"])
non_member_indices = non_member_indices[:30000]
# Create combined data using only member and non-member indices
combined_indices = np.concatenate([member_indices, non_member_indices])
all_data = torch.tensor(features[combined_indices], dtype=torch.float32, device=device)
all_labels = torch.tensor(labels[combined_indices], dtype=torch.long, device=device)

# Create membership labels (1 for members, 0 for non-members)
is_member = np.zeros(len(combined_indices), dtype=int)
is_member[:len(member_indices)] = 1

logger.info(f"Testing with reduced dataset: {len(combined_indices)} samples ({len(member_indices)} members + {len(non_member_indices)} non-members)")

# Load snapshots
combined_snapshots_path = f"combined_model_snapshots{num_epochs}.pth"
combined_snapshots = torch.load(combined_snapshots_path)
logger.info(f"Loaded combined snapshots for epochs: {list(combined_snapshots.keys())}")

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

def compute_attack_vector(batch_indices, all_data, is_member, snapshots):
    attack_vectors = []
    labels = []
    num_epochs = 5
    learning_rate = .001
    
    for idx in batch_indices:
        feature = all_data[idx]
        label = is_member[idx]
        bias_changes = []
        
        for snapshot_id, snapshot in snapshots.items():
            model = create_model().to(device)
            model.load_state_dict(snapshot)
            model.train()
            
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            bias_before = model[-2].bias.detach().cpu().numpy()
            
            feature_tensor = feature.unsqueeze(0)
            target = torch.argmax(all_labels[idx]).unsqueeze(0)
            
            for _ in range(num_epochs):
                optimizer.zero_grad()
                output = model(feature_tensor)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
            
            bias_after = model[-2].bias.detach().cpu().numpy()
            bias_change = bias_after - bias_before
            bias_changes.append(bias_change)
        
        bias_deltas = [bias_changes[i+1] - bias_changes[i] for i in range(len(bias_changes)-1)]
        attack_vector = np.concatenate(bias_deltas)
        #logger.info(f"Attack vector for sample {idx}: {attack_vector}")

        #logger.info(f"Attack label for sample {idx}: {label}")
        attack_vectors.append(attack_vector)
        labels.append(label)
    
    return attack_vectors, labels

# Process in batches
batch_size = 64
total_samples = len(member_indices) + len(non_member_indices)
num_batches = (total_samples + batch_size - 1) // batch_size  

logger.info(f"Processing {total_samples} samples in {num_batches} batches")

batch_indices_list = [range(i * batch_size, min((i + 1) * batch_size, total_samples)) 
                     for i in range(num_batches)]

attack_vectors = []
labels = []

# Process each batch
for i, batch_indices in enumerate(batch_indices_list):
    logger.info(f"Processing batch {i + 1} / {num_batches}...")
    batch_vectors, batch_labels = compute_attack_vector(
        batch_indices,
        all_data,
        is_member,
        combined_snapshots
    )
    attack_vectors.extend(batch_vectors)
    labels.extend(batch_labels)
    gc.collect()

# After processing, create visualization
attack_vectors = np.array(attack_vectors)
labels = np.array(labels)

# Calculate mean magnitude of attack vectors
magnitudes = np.mean(np.abs(attack_vectors), axis=1)

# Create plot
plt.figure(figsize=(12, 6))
sns.histplot(data={'Member': magnitudes[labels == 1], 
                   'Non-member': magnitudes[labels == 0]}, 
             common_norm=True, 
             stat='density',
             alpha=0.6)
plt.title('Distribution of Attack Vector Magnitudes')
plt.xlabel('Mean Magnitude')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)
plt.savefig(f'attack_vector_distribution{num_epochs}.png')
plt.close()

logger.info(f"Distribution plot saved as 'attack_vector_distribution{num_epochs}.png'")

# Save the test results
np.save(f"attack_vectors{num_epochs}.npy", attack_vectors)
np.save(f"attack_labels{num_epochs}.npy", labels)
logger.info("Test vectors and labels saved")