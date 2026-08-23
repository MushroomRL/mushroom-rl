from .air_hockey import AirHockeyHit, AirHockeyDefend, AirHockeyPrepare, AirHockeyRepel
from .locomotion import Hopper, Walker2D, HalfCheetah, Ant
from .manipulation import Reach, Push, Pick, PegInsertion
from .ball_in_a_cup import BallInACup

BallInACup.register()
AirHockeyHit.register()
AirHockeyDefend.register()
AirHockeyPrepare.register()
AirHockeyRepel.register()
Hopper.register()
Walker2D.register()
HalfCheetah.register()
Ant.register()
Reach.register()
Push.register()
Pick.register()
PegInsertion.register()
