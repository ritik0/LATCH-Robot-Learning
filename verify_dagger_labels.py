import numpy as np

old_obs = np.load("observations.npy")

all_obs = np.load("dagger_observations.npy")
all_actions = np.load("dagger_actions.npy")

new_obs = all_obs[len(old_obs):]
saved_actions = all_actions[len(old_obs):]

def expert_policy(obs):
    dx = obs[8]
    dy = obs[9]

    qd1 = obs[6]
    qd2 = obs[7]

    kp = 1.0
    kd = 0.05

    action1 = -kp * dx - kd * qd1
    action2 = -kp * dy - kd * qd2

    action = np.array(
        [action1, action2], dtype = np.float32)

    return np.clip(action, -1.0, 1.0)

recomputed_actions = np.array(
    [expert_policy(obs) for obs in new_obs]
)

difference = saved_actions - recomputed_actions

print("Saved action min:")
print(saved_actions.min(axis=0))

print("Saved action max:")
print(saved_actions.max(axis=0))


print("\nRecomputed action min:")
print(recomputed_actions.min(axis=0))

print("Recomputed action max:")
print(recomputed_actions.max(axis=0))


print("\nMean absolute difference:")
print(np.mean(np.abs(difference), axis=0))


print("\nFirst 10 saved actions:")
print(saved_actions[:10])

print("\nFirst 10 recomputed actions:")
print(recomputed_actions[:10])