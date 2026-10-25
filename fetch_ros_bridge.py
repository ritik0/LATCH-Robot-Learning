import gymnasium as gym
import numpy as np
import gymnasium_robotics

import rclpy

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from fetch_push_expert import FetchPushExpert

gym.register_envs(gymnasium_robotics)

class FetchSimBridge(Node):
    def __init__(self):
        super().__init__("fetch_sim_bridge")

        self.publisher = self.create_publisher(
            Float64MultiArray,
            "/latch/observation",
            10
        )

        self.subscription = self.create_subscription(
            Float64MultiArray,
            "/latch/action",
            self.action_callback,
            10
        )

        self.env = gym.make(
            "FetchPushDense-v4"
        )

        self.obs, self.info = self.env.reset(
            seed=0
        )

        self.latest_action = np.zeros(
            4, dtype = np.float32
        )
        self.timer = self.create_timer(
            0.05,
            self.step_simulation
        )

        self.get_logger().info(
            "Fetch simulation bridge started"
        )
    
    def action_callback(self, msg):
        if len(msg.data) != 4:
            self.get_logger().warning(
                "Received invalid action size"
            )
            return
        
        self.latest_action = np.array(
            msg.data,
            dtype=np.float32
        )
    
    def step_simulation(self):
        policy_input = np.concatenate(
            [self.obs["observation"],
            self.obs["desired_goal"]
            ]
        )

        message = Float64MultiArray()

        message.data = policy_input.tolist()

        self.publisher.publish(message)
        
        self.obs, reward, terminated, truncated, info = (
            self.env.step(self.latest_action)
        )
        if terminated or truncated:
            self.obs, self.info = self.env.reset()
            self.latest_action = np.zeros(
                4,
                dtype=np.float32
            )

    def close(self):
        self.env.close()

def main():
    rclpy.init()
    node = FetchSimBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()