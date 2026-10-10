import gymnasium as gym
import numpy as np


def simple_policy(obs):
    dx = obs[8]
    dy = obs[9]

    qd1 = obs[6]
    qd2 = obs[7]

    kp = 1.0
    kd = 0.05

    action1 = -kp * dx - kd * qd1
    action2 = -kp * dy - kd * qd2

    action = np.array([action1, action2], dtype=np.float32)

    return np.clip(action, -1.0, 1.0)


env = gym.make("Reacher-v5")

observations = []
actions = []

for episode in range(20):
    obs, info = env.reset()

    while True:
        action = simple_policy(obs)

        observations.append(obs.copy())
        actions.append(action.copy())

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

env.close()

observations = np.array(observations, dtype=np.float32)
actions = np.array(actions, dtype=np.float32)

print("Observations shape:", observations.shape)
print("Actions shape:", actions.shape)

np.save("observations.npy", observations)
np.save("actions.npy", actions)