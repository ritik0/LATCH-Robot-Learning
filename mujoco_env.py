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

def random_policy(obs):
    return env.action_space.sample()

def run_episode(env, policy):
    obs, info = env.reset()
    episode_return = 0.0

    while True:
        action = policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        episode_return += reward

        if terminated or truncated:
            break

    return episode_return

def evaluate_policy(env, policy, episodes = 20):
    returns = []

    for _ in range(episodes):
        score = run_episode(env, policy)
        returns.append(score)
    return np.mean(returns), np.std(returns)


random_mean, random_std = evaluate_policy(env, random_policy)
simple_mean, simple_std = evaluate_policy(env, simple_policy)

print("Random Policy:", random_mean, "+/-", random_std)
print("Simple Policy:", simple_mean, "+/-", simple_std)

score = run_episode(env, simple_policy)
print("Episode return:", score)

env.close()