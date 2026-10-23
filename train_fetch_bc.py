import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, random_split

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("using device:", device)

observations = np.load("fetch_observations.npy")
actions = np.load("fetch_actions.npy")

print("Observations:", observations.shape)
print("Actions:", actions.shape)

obs_tensor = torch.tensor(observations, dtype = torch.float32)
action_tensor = torch.tensor(actions, dtype = torch.float32)

dataset = TensorDataset(obs_tensor, action_tensor)
train_size = int(0.8 * len(dataset))
val_size = (len(dataset) - train_size)

train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator = torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 128, shuffle = False)

class FetchBCPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(28,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,4)
        )

    def forward(self, x):
        return self.network(x)

policy = FetchBCPolicy().to(device)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(
    policy.parameters(),
    lr = 0.001
)

epochs = 50

best_val_loss = float("inf")

for epoch in range(epochs):
    policy.train()
    train_loss = 0.0

    for batch_obs,  batch_actions in train_loader: 

        batch_obs = batch_obs.to(device)
        batch_actions = batch_actions.to(device)

        predictions = policy(batch_obs)
        loss = loss_function(predictions, batch_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    average_train_loss = (train_loss / len(train_loader))

    policy.eval()
    val_loss = 0.0

    with torch.no_grad():

        for batch_obs, batch_actions in val_loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            predictions = policy(batch_obs)
            loss = loss_function(predictions, batch_actions)
            val_loss += loss.item()

    average_val_loss = ( val_loss / len(val_loader))

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss
        torch.save(policy.state_dict(), "fetch_bc_policy.pt")

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch {epoch + 1}: "
            f"Train = {average_train_loss:.6f}, "
            f"Val = {average_val_loss:.6f}"
        )


print(
    "Best validation loss:",
    best_val_loss
)

print("Model saved")