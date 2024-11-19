import torch
import torch.nn as nn

# Load model snapshots from specified rounds
model_snapshots = {
    "round_1": torch.load("client_4_global_snapshots_round_1.pth"),
    "round_2": torch.load("client_4_global_snapshots_round_2.pth"),
    "round_3": torch.load("client_4_global_snapshots_round_3.pth"),
    "round_4": torch.load("client_4_global_snapshots_round_4.pth"),
    "round_5": torch.load("client_4_global_snapshots_round_5.pth"),
    "round_6": torch.load("client_4_global_snapshots_round_6.pth"),
    "round_7": torch.load("client_4_global_snapshots_round_7.pth"),
    "round_8": torch.load("client_4_global_snapshots_round_8.pth"),
    "round_9": torch.load("client_4_global_snapshots_round_9.pth"),
    "round_10": torch.load("client_4_global_snapshots_round_10.pth")
}
selected_epochs = [
    ("round_1", 10), ("round_2", 10),
    ("round_3", 10), ("round_4", 10),
    ("round_5", 10), ("round_6", 10),
      ("round_7", 10), ("round_8", 10),
    ("round_9", 10), ("round_10", 10)
]
    # Assuming you have a model to load this into
def create_model():
    model = nn.Sequential(
        nn.Linear(6169, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 100)
    )
    for param in model.parameters():
        param.requires_grad = True
    return model
# Extract the selected epochs data from the model snapshots
selected_data = {}
for round_name, epoch in selected_epochs:
    if round_name in model_snapshots:
        model_snapshot = model_snapshots[round_name]
        selected_data[f"{round_name}_epoch_{epoch}"] = model_snapshot[f"epoch_{epoch}"]

# Save selected data to a .pth file
torch.save(selected_data, "combined_model_snapshots.pth")
# Load the saved model snapshots

