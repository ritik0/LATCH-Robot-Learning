import gymnasium as gym
import gymnasium_robotics

from fetch_push_expert import FetchPushExpert

gym.register_envs(gymnasium_robotics)

env = gym.make(
    "FetchPushDense-v4"
)

expert = FetchPushExpert()
expert.reset()

obs, info = env.reset(seed = 0)

episode_return = 0.0

for step in range(100):
    action = expert.get_action(obs)
    obs, reward, terminated, truncated, info = env.step(
        action
    )

    episode_return += reward
    if terminated or truncated: 
        break
    
print("Episode return:", episode_return)
print("Final object position:", obs["achieved_goal"])
print("Goal position:", obs["desired_goal"])

env.close()