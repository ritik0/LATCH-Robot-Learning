import gymnasium as gym
import gymnasium_robotics
import numpy as np

from fetch_push_expert import FetchPushExpert

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchPushDense-v4")

expert = FetchPushExpert()

observations = []
actions = []

episodes = 50

for episode in range(episodes):
    obs, info = env.reset(seed = episode)
    expert.reset()

    while True:
        policy_input = np.concatenate(
            [
                obs["observation"],
                obs["desired_goal"]
            ]
        )
        action = expert.get_action(obs)

        observations.append(policy_input.copy())
        actions.append(action.copy())

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated: 
            break
    print (f"Episode {episode +1}/{episodes}")

env.close()

observations = np.array(
    observations, dtype = np.float32
)
actions = np.array(
    actions, dtype = np.float32
)

print("\n Dataset collected")

print("Observations:", observations.shape)
print("Actions:", actions.shape)

np.save("fetch_observations.npy", observations)
np.save("fetch_actions.npy", actions)

print("Dataset Saved")