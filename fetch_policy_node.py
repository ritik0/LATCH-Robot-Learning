import numpy as np
import torch
import torch.nn as nn
import rclpy

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class FetchBCPolicy(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128,4)
        )

    def forward(self, x):
        return self.network(x)

class FetchPolicyNode(Node):
    def __init__(self):
        super().__init__("fetch_policy_node")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy = FetchBCPolicy().to(
            self.device
        )

        self.policy.load_state_dict(
            torch.load(
                "fetch_bc_policy.pt",
                map_location=self.device
            )
        )

        self.policy.eval()

        self.subscription = self.create_subscription(
            Float64MultiArray,
            "/latch/observation",
            self.observation_callback,
            10
        )

        self.publisher = self.create_publisher(
            Float64MultiArray,
            "/latch/policy_action",
            10
        )

        self.get_logger().info(
            f"Policy node started on {self.device}"
        )

    def observation_callback(self, msg):
        if len(msg.data) != 28:

            self.get_logger().warning(
                f"Expected 28 values got {len(msg.data)}"
            )

            return
        observation = np.array(
            msg.data,
            dtype=np.float32
        )

        observation_tensor = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            action_tensor = self.policy(
                observation_tensor
            )

        action = (
            action_tensor
            .squeeze(0)
            .cpu()
            .numpy()
        )

        action = np.clip(
            action, -1.0 ,1.0
        )

        message = Float64MultiArray()
        message.data = action.tolist()
        self.publisher.publish(
            message
        )

def main():
    rclpy.init()
    node = FetchPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally: 
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
