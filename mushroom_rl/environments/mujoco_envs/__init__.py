from .ball_in_a_cup import BallInACup
from .air_hockey import AirHockeyHit, AirHockeyDefend, AirHockeyPrepare, AirHockeyRepel
from .ant import Ant
from .half_cheetah import HalfCheetah
from .hopper import Hopper
from .walker_2d import Walker2D
from .reach import Reach
from .push import Push
from .pick import Pick
from .peg_insertion import PegInsertion

BallInACup.register()
AirHockeyHit.register()
AirHockeyDefend.register()
AirHockeyPrepare.register()
AirHockeyRepel.register()
Ant.register()
HalfCheetah.register()
Hopper.register()
Walker2D.register()
Reach.register()
Push.register()
Pick.register()
PegInsertion.register()
