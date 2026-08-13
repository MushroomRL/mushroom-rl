try:
    Atari = None
    from .atari import Atari

    Atari.register()
except ImportError:
    pass

try:
    Gymnasium = None
    from .gymnasium_env import Gymnasium

    Gymnasium.register()
except ImportError:
    pass

try:
    DMControl = None
    from .dm_control_env import DMControl

    DMControl.register()
except ImportError:
    pass

try:
    MiniGrid = None
    MiniGridRGB = None
    from .minigrid_env import MiniGrid, MiniGridRGB

    MiniGrid.register()
    MiniGridRGB.register()
except ImportError:
    pass

try:
    iGibson = None
    from .igibson_env import iGibson

    iGibson.register()
except ImportError:
    import logging

    logging.disable(logging.NOTSET)

try:
    Habitat = None
    from .habitat_env import Habitat

    Habitat.register()
except ImportError:
    pass

try:
    MuJoCo = None
    from .mujoco import MuJoCo, MultiMuJoCo
    from .mujoco_envs import *
except ImportError:
    pass

try:
    OmniIsaacGymEnv = None
    from .omni_isaac_gym_env import OmniIsaacGymEnv
except ImportError:
    pass

try:
    PyBullet = None
    from .pybullet import PyBullet
    from .pybullet_envs import *
except ImportError:
    pass

try:
    IsaacSim = None
    from .isaacsim_env import IsaacSim
except ImportError:
    pass
