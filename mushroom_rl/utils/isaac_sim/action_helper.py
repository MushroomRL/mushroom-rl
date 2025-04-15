from enum import Enum

class ActionType(Enum):
    EFFORT = "joint_efforts"
    POSITION = "joint_positions"
    VELOCITY = "joint_velocities"