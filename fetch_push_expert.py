import numpy as np

class FetchPushExpert:

    def __init__(self):
        self.approach_distance = 0.06
        self.position_gain = 8.0

        self.hover_height = 0.08
        self.phase = "HOVER"
    
    def reset(self):
        self.phase = "HOVER"

    def get_action(self, obs):
        state = obs["observation"]
        gripper_pos = state[0:3]
        object_pos = state[3:6]
        goal_pos = obs["desired_goal"]



        direction = goal_pos[:2] - object_pos[:2]

        distance = np.linalg.norm(direction)

        if distance > 1e-6:
            direction = direction / distance
        else: 
            direction = np.zeros(2)

        behind_pos = object_pos.copy()

        behind_pos[:2] -= (
            self.approach_distance * direction
        )

        if self.phase == "HOVER":
            target = behind_pos.copy()
            target[2] += self.hover_height
            error = target - gripper_pos

            if np.linalg.norm(error) < 0.025:
                self.phase = "DESCEND"
                print("Switching to DESCEND")

        elif self.phase == "DESCEND":
            target = behind_pos.copy()
            error = target - gripper_pos

            if np.linalg.norm(error) < 0.02:
                self.phase = "PUSH"
                print("Switching to PUSH")
            
        else:
            target = goal_pos.copy()
            target[2] = object_pos[2]
            error = target - gripper_pos

        action_xyz = (
            self.position_gain * error
        )

        action_xyz = np.clip(action_xyz, -1.0, 1.0)

        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], 0.0], dtype = np.float32)

        return action