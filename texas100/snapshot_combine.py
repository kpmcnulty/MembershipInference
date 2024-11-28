import torch
import torch.nn as nn
### COMBINE SNAPSHOTS TO 1 FILE FOR EASIER LOCAL ATTACK
## Also includes a debugging check to see if the models are different.
selected_epochs = [5, 10, 20, 25, 30, 35, 45, 50, 60, 85]
# Path to store the combined snapshots
combined_snapshots_path = "combined_snapshots.npy"

# Initialize dictionary to store combined snapshots
combined_snapshots = {}
snapshot_differences = []

for i, epoch in enumerate(selected_epochs):
    snapshot_file = f"snaphot__{epoch}.pth"
    try:
        snapshot = torch.load(snapshot_file)
        combined_snapshots[epoch] = snapshot
        print(f"Loaded snapshot for epoch {epoch}")
        
        # Compare with the previous snapshot (if not the first one)
        if i > 0:
            prev_epoch = selected_epochs[i - 1]
            prev_snapshot = combined_snapshots[prev_epoch]
            
            # Check differences between snapshots
            total_diff = 0
            for key in snapshot.keys():
                if not torch.equal(snapshot[key], prev_snapshot[key]):
                    diff = (snapshot[key] - prev_snapshot[key]).abs().sum().item()
                    total_diff += diff
            snapshot_differences.append((prev_epoch, epoch, total_diff))
            print(f"Difference between epoch {prev_epoch} and {epoch}: {total_diff}")
        else:
            print(f"Epoch {epoch} is the first snapshot, no comparison possible.")

    except FileNotFoundError:
        print(f"Snapshot file not found: {snapshot_file}")

# Save the combined snapshots
torch.save(combined_snapshots, "combined_model_snapshots.pth")
print("Combined snapshots saved to combined_model_snapshots.pth")

# Summary of differences
print("\nSummary of Differences Between Snapshots:")
for prev_epoch, curr_epoch, diff in snapshot_differences:
    print(f"Epoch {prev_epoch} -> Epoch {curr_epoch}: Total Parameter Difference = {diff}")
