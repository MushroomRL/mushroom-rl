__extras__ = []

from .environment import Environment, MDPInfo
try:
    Atari = None
    from .atari import Atari
    __extras__.append('Atari')
except ImportError:
    pass

try:
    Gym = None
    from .gym_env import Gym
    __extras__.append('Gym')
except ImportError:
    pass

try:
    DMControl = None
    from .dm_control_env import DMControl
    __extras__.append('DMControl')
except ImportError:
    pass

try:
    Mujoco = None
    from .mujoco import MuJoCo
    __extras__.append('Mujoco')
except ImportError:
    pass

try:
    PyBullet = None
    from .pybullet import PyBullet
    __extras__.append('PyBullet')
except ImportError:
    pass

from .car_on_hill import CarOnHill
from .generators.simple_chain import generate_simple_chain
from .grid_world import GridWorld, GridWorldVanHasselt
from .finite_mdp import FiniteMDP
from .inverted_pendulum import InvertedPendulum
from .cart_pole import CartPole
from .puddle_world import PuddleWorld
from .ship_steering import ShipSteering
from .lqr import LQR

__all__ = ['CarOnHill', 'Environment', 'MDPInfo',
           'FiniteMDP', 'InvertedPendulum', 'CartPole',
           'GridWorld', 'generate_simple_chain', 'GridWorldVanHasselt',
           'PuddleWorld', 'ShipSteering', 'LQR'] + __extras__
