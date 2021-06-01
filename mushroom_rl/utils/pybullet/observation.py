from enum import Enum


class PyBulletObservationType(Enum):
    """
    An enum indicating the type of data that should be added to the observation
    of the environment, can be Joint-/Body-/Site- positions and velocities.

    """
    __order__ = "BODY_POS BODY_LIN_VEL BODY_ANG_VEL JOINT_POS JOINT_VEL LINK_POS LINK_LIN_VEL LINK_ANG_VEL CONTACT_FLAG"
    BODY_POS = 0
    BODY_LIN_VEL = 1
    BODY_ANG_VEL = 2
    JOINT_POS = 3
    JOINT_VEL = 4
    LINK_POS = 5
    LINK_LIN_VEL = 6
    LINK_ANG_VEL = 7
    CONTACT_FLAG = 8