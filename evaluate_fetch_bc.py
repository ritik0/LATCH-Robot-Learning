import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn

gym.register_envs(gymnasium_robotics)


class FetchBCPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        return self.network(x)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

policy = FetchBCPolicy().to(device)

policy.load_state_dict(
    torch.load(
        "fetch_bc_policy.pt",
        map_location=device
    )
)

policy.eval()


def get_action(obs):

    policy_input = np.concatenate(
        [
            obs["observation"],
            obs["desired_goal"]
        ]
    )

    obs_tensor = torch.tensor(
        policy_input,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        action = policy(obs_tensor)

    action = action.cpu().numpy()

    return np.clip(
        action,
        -1.0,
        1.0
    )


env = gym.make("FetchPushDense-v4")

returns = []

for episode in range(10):

    obs, info = env.reset(seed=100 + episode)

    episode_return = 0.0

    while True:

        action = get_action(obs)

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        episode_return += reward

        if terminated or truncated:
            break

    returns.append(episode_return)

    print(
        f"Episode {episode + 1}: "
        f"{episode_return:.3f}"
    )


print(
    "\nMean return:",
    np.mean(returns)
)

print(
    "Std:",
    np.std(returns)
)

env.close()