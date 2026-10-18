import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

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

policy.load_state_dict(
    torch.load("bc_policy.pt")
)

policy.eval()

def bc_policy(obs):
    obs_tensor = torch.tensor(
        obs, dtype = torch.float32
    )
    with torch.no_grad():
        action_tensor = policy(obs_tensor)
    action = action_tensor.numpy()

    return np.clip(action, -1.0, 1.0)

def expert_policy(obs):
    dx = obs[8]
    dy = obs[9]
    qd1 = obs[6]
    qd2 = obs[7]

    kp = 1.0
    kd = 0.05

    action1 = -kp * dx - kp * qd1
    action2 = -kp * dy - kd * qd2

    action = np.array(
        [action1, action2],
        dtype = np.float32
    )

    return np.clip(action, -1.0, 1.0)

env = gym.make("Reacher-v5")

new_observations = []
new_actions = []

episodes = 10

for episode in range(episodes):
    obs, info = env.reset(
        seed = 100 + episode
    )

    while True:
        expert_action = expert_policy(obs)

        new_observations.append(
            obs.copy()
        )
        new_actions.append(
            expert_action.copy()
        )

        action = bc_policy(obs)
        obs, reward, terminated, truncated, info = env.step(
            action
        )

        if terminated or truncated:
            break
env.close()

new_observations = np.array(new_observations)
new_actions = np.array(new_actions)

print(
    "New observations:",
    new_observations.shape
)

print(
    "New expert labels:",
    new_actions.shape
)

old_observations = np.load("observations.npy")
old_actions = np.load("actions.npy")

dagger_observations = np.concatenate(
    [old_observations, new_observations], axis = 0
)
dagger_actions = np.concatenate(
    [old_actions, new_actions], axis = 0
)

print("Original dataset:", old_observations.shape)

print("DAgger dataset:", dagger_observations.shape)

np.save("dagger_observations.npy", dagger_observations)

np.save("dagger_actions.npy", dagger_actions)

print("DAgger dataset saved")