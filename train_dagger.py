import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset, DataLoader, random_split
)

# Dataset Loading 
observations = np.load("dagger_observations.npy")
actions = np.load("dagger_actions.npy")

print("Observations:", observations.shape)
print("Actions:", actions.shape)

#Convert to tensors

obs_tensor = torch.tensor(
    observations, dtype = torch.float32
)

action_tensor = torch.tensor(
    actions, dtype = torch.float32
)

#Dataset split

dataset = TensorDataset(
    obs_tensor, action_tensor
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
                    dataset,
                    [train_size, val_size],
                    generator = torch.Generator().manual_seed(42)
)
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size = 64, 
    shuffle = True
)
val_loader = DataLoader(
    val_dataset,
    batch_size = 64,
    shuffle = False
)

#Define policy

class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
    def forward(self, x):
        return self.network(x)
policy = BCPolicy()

#Loss and Optimizer

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(
    policy.parameters(),
    lr = 0.001
)

#Training

epochs = 50

best_val_loss = float("inf")

for epoch in range(epochs):
    policy.train()
    total_train_loss = 0.0

    for batch_obs, batch_actions in train_loader:
        predicted_actions = policy(batch_obs)

        loss = loss_function(
            predicted_actions, 
            batch_actions
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
    average_train_loss = (
        total_train_loss / len(train_loader)
    )

    policy.eval()

    total_val_loss = 0.0

    with torch.no_grad():
        for batch_obs, batch_actions in val_loader:
            predicted_actions = policy(batch_obs)

            loss = loss_function(
                predicted_actions,
                batch_actions
            )

            total_val_loss += loss.item()
    average_val_loss = (total_val_loss / len(val_loader))

    #Save best model

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss

        torch.save(
            policy.state_dict(),
            "dagger_policy.pt"
        )
    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"Train loss = {average_train_loss:.6f}, "
            f"Val loss = {average_val_loss:.6f}"
        )


print("Best Dagger model saved")
print("Best validation loss:", best_val_loss)