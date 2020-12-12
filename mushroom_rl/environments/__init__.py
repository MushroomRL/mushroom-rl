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

from .generators.simple_chain import generate_simple_chain

from .car_on_hill import CarOnHill
from .cart_pole import CartPole
from .finite_mdp import FiniteMDP
from .grid_world import GridWorld, GridWorldVanHasselt
from .inverted_pendulum import InvertedPendulum
from .lqr import LQR
from .puddle_world import PuddleWorld
from .segway import Segway
from .ship_steering import ShipSteering


__all__ = ['Environment', 'MDPInfo',  'generate_simple_chain',
           'CarOnHill',  'CartPole', 'FiniteMDP',
           'GridWorld', 'GridWorldVanHasselt', 'InvertedPendulum',
           'LQR', 'PuddleWorld', 'Segway',
           'ShipSteering' ] + __extras__
