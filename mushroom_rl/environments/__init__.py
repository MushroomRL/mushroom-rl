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
    from .minigrid_env import MiniGrid
    MiniGrid.register()
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
    IsaacEnv = None
    from .isaac_env import IsaacEnv
except ImportError:
    pass

try:
    PyBullet = None
    from .pybullet import PyBullet
    from .pybullet_envs import *
except ImportError:
    pass

from .generators.simple_chain import generate_simple_chain

from .car_on_hill import CarOnHill
CarOnHill.register()

from .cart_pole import CartPole
CartPole.register()

from .finite_mdp import FiniteMDP
FiniteMDP.register()

from .grid_world import GridWorld, GridWorldVanHasselt
GridWorld.register()
GridWorldVanHasselt.register()

from .inverted_pendulum import InvertedPendulum
InvertedPendulum.register()

from .lqr import LQR
LQR.register()

from .puddle_world import PuddleWorld
PuddleWorld.register()

from .segway import Segway
Segway.register()

from .ship_steering import ShipSteering
ShipSteering.register()
