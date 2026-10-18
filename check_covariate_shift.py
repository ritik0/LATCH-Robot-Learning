import numpy as np
import torch
import torch.nn as nn

class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self, x):
        return self.network(x)

bc_policy = BCPolicy()
bc_policy.load_state_dict(
    torch.load("bc_policy.pt")
)
bc_policy.eval()

dagger_policy = BCPolicy()
dagger_policy.load_state_dict(
    torch.load("dagger_policy.pt")
)
dagger_policy.eval()

old_obs = np.load("observations.npy")
old_actions = np.load("actions.npy")

all_obs = np.load("dagger_observations.npy")
all_actions = np.load("dagger_actions.npy")

new_obs = all_obs[len(old_obs):]
new_actions = all_actions[len(old_actions):]

def calculate_mse(policy, observations, expert_actions):
    obs_tensor = torch.tensor(
        observations,
        dtype = torch.float32
    )

    action_tensor = torch.tensor(
        expert_actions, 
        dtype = torch.float32
    )

    with torch.no_grad():
        predictions = policy(obs_tensor)
    mse = torch.mean(
        (predictions - action_tensor) ** 2
    )

    return mse.item()

bc_old_mse = calculate_mse(bc_policy, old_obs, old_actions)
bc_new_mse = calculate_mse(bc_policy, new_obs, new_actions)

dagger_old_mse = calculate_mse(dagger_policy, old_obs, old_actions)
dagger_new_mse = calculate_mse(dagger_policy, new_obs, new_actions)

print("BC policy")
print("Original expert states:", bc_old_mse)
print("BC-visited states:", bc_new_mse)

print()

print("DAgger policy")
print("Original expert states:", dagger_old_mse)
print("BC-visited states:", dagger_new_mse)