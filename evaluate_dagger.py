import gymnasium as gym 
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
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

bc_model = BCPolicy()

bc_model.load_state_dict(
    torch.load("bc_policy.pt")
)

bc_model.eval()

dagger_model = BCPolicy()

dagger_model.load_state_dict(
    torch.load("dagger_policy.pt")
)

dagger_model.eval()

def bc_policy(obs):
    obs_tensor = torch.tensor(obs, dtype = torch.float32)
    with torch.no_grad():
        action_tensor = bc_model(obs_tensor)
    action = action_tensor.numpy()

    return np.clip(action, -1.0, 1.0)

def dagger_policy(obs):
    obs_tensor = torch.tensor(
        obs,
        dtype=torch.float32
    )
    with torch.no_grad():
        action_tensor = dagger_model(obs_tensor)
    action = action_tensor.numpy()

    return np.clip(action, -1.0, 1.0)

def expert_policy(obs):
    dx = obs[8]
    dy = obs[9]

    qd1 = obs[6]
    qd2 = obs[7]

    kp = 1.0
    kd = 0.05

    action1 = -kp * dx - kd * qd1
    action2 = -kp * dy - kd * qd2

    action = np.array([action1, action2], dtype = np.float32)

    return np.clip(action, -1.0, 1.0)

def run_episode(env, policy_fn, seed):
    obs, info = env.reset(seed = seed)
    env.action_space.seed(seed)

    episode_return = 0.0

    while True:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += reward

        if terminated or truncated:
            break
    return episode_return

def evaluate_policy(env, policy_fn, seeds):
    returns = []

    for seed in seeds:
        score = run_episode(env, policy_fn, seed)
        returns.append(score)

    return(np.mean(returns), np.std(returns))

env = gym.make("Reacher-v5")
seeds = list(range(20))

expert_mean, expert_std = evaluate_policy(env, expert_policy, seeds)
bc_mean, bc_std = evaluate_policy(env, bc_policy, seeds)
dagger_mean, dagger_std = evaluate_policy(env, dagger_policy, seeds)

print(
    "Expert Policy:",
    expert_mean,
    "+/-",
    expert_std
)

print(
    "BC Policy:",
    bc_mean,
    "+/-",
    bc_std
)

print(
    "DAgger Policy:",
    dagger_mean,
    "+/-",
    dagger_std
)


env.close()