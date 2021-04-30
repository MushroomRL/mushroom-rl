__extras__ = []

from .environment import Environment, MDPInfo
try:
    Atari = None
    from .atari import Atari
    __extras__.append('Atari')
    Atari.register()
except ImportError:
    pass

try:
    Gym = None
    from .gym_env import Gym
    __extras__.append('Gym')
    Gym.register()
except ImportError:
    pass

try:
    DMControl = None
    from .dm_control_env import DMControl
    __extras__.append('DMControl')
    DMControl.register()
except ImportError:
    pass

try:
    MuJoCo = None
    from .mujoco import MuJoCo
    __extras__.append('MuJoCo')
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


__all__ = ['Environment', 'MDPInfo',  'generate_simple_chain',
           'CarOnHill',  'CartPole', 'FiniteMDP',
           'GridWorld', 'GridWorldVanHasselt', 'InvertedPendulum',
           'LQR', 'PuddleWorld', 'Segway',
           'ShipSteering' ] + __extras__
