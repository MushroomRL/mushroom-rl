from mushroom_rl.environments.pybullet_envs.air_hockey.hit import AirHockeyHit
from mushroom_rl.environments.pybullet_envs.air_hockey.defend import AirHockeyDefend
from mushroom_rl.environments.pybullet_envs.air_hockey.prepare import AirHockeyPrepare
from mushroom_rl.environments.pybullet_envs.air_hockey.repelle import AirHockeyRepelle
from mushroom_rl.environments.pybullet_envs.air_hockey.defend_hit import AirHockeyDefendHit
from mushroom_rl.environments.pybullet_envs.air_hockey.double import AirHockeyDouble
from mushroom_rl.environments.pybullet_envs.air_hockey.simple_double import AirHockeySimpleDouble
from mushroom_rl.environments.pybullet_envs.air_hockey.return_init import AirHockeyReturn



AirHockeyHit.register()
AirHockeyDefend.register()
AirHockeyPrepare.register()
AirHockeyRepelle.register()
AirHockeyDefendHit.register()
AirHockeyDouble.register()
AirHockeyReturn.register()
