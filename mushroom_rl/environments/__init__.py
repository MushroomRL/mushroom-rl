try:
    from .atari import Atari
    Atari.register()
except ImportError:
    Atari = None

try:
    from .gymnasium_env import Gymnasium
    Gymnasium.register()
except ImportError:
    Gymnasium = None

try:
    from .dm_control_env import DMControl
    DMControl.register()
except ImportError:
    DMControl = None

try:
    from .minigrid_env import MiniGrid, MiniGridRGB
    MiniGrid.register()
    MiniGridRGB.register()
except ImportError:
    MiniGrid = None
    MiniGridRGB = None

try:
    from .mujoco import MuJoCo, MultiMuJoCo
    from .mujoco_envs import *  # noqa: F401,F403
except ImportError:
    MuJoCo = None
    MultiMuJoCo = None

try:
    from .mujoco_warp import MuJoCoWarp
    from .mujoco_warp_envs import *  # noqa: F401,F403
except ImportError:
    MuJoCoWarp = None

try:
    from .pybullet import PyBullet
    from .pybullet_envs import *  # noqa: F401,F403
except ImportError:
    PyBullet = None

from .car_on_hill import CarOnHill
from .cart_pole import CartPole
from .finite_mdp import FiniteMDP
from .grid_world import GridWorld
from .grid_world_van_hasselt import GridWorldVanHasselt
from .inverted_pendulum import InvertedPendulum
from .lqr import LQR
from .puddle_world import PuddleWorld
from .segway import Segway
from .ship_steering import ShipSteering
from .simple_chain import SimpleChain
from .taxi import Taxi

CarOnHill.register()
CartPole.register()
FiniteMDP.register()
GridWorld.register()
GridWorldVanHasselt.register()
InvertedPendulum.register()
LQR.register()
PuddleWorld.register()
Segway.register()
ShipSteering.register()
SimpleChain.register()
Taxi.register()
