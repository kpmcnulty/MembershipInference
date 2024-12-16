import torch
import torch.nn as nn
### COMBINE SNAPSHOTS TO 1 FILE FOR EASIER LOCAL ATTACK
## Also includes a debugging check to see if the models are different.
selected_epochs = [5, 10, 20, 25, 30, 35, 45, 50, 60, 85]
num_epochs = 5 #########
# Path to store the combined snapshots
combined_snapshots_path = f"combined_model_snapshots{num_epochs}.pth"

# Initialize dictionary to store combined snapshots
combined_snapshots = {}


for i, epoch in enumerate(selected_epochs):
    snapshot_file = f"models{num_epochs}/local_{epoch}.pth"
    snapshot = torch.load(snapshot_file)
    combined_snapshots[epoch] = snapshot
    print(f"Loaded snapshot for epoch {epoch}")

# Save the combined snapshots
torch.save(combined_snapshots, f"combined_model_snapshots{num_epochs}.pth")
print(f"Combined snapshots saved to combined_model_snapshots{num_epochs}.pth")