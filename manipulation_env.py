import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchPushDense-v4")

print("Observation space:")
print(env.observation_space)

print("\nAction space:")
print(env.action_space)

obs, info = env.reset(seed = 0)

print("\nObservation type:")
print(type(obs))
print("\nObservation keys:")
print(obs.keys())
print("\nObservation:")
print(obs)
print("\nInfo:")
print(info)

env.close()