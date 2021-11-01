from mushroom_rl.environments.pybullet_envs.air_hockey.hit import AirHockeyHit
from mushroom_rl.environments.pybullet_envs.air_hockey.defend import AirHockeyDefend
from mushroom_rl.environments.pybullet_envs.air_hockey.prepare import AirHockeyPrepare
from mushroom_rl.environments.pybullet_envs.air_hockey.repelle import AirHockeyRepelle


AirHockeyHit.register()
AirHockeyDefend.register()
AirHockeyPrepare.register()
AirHockeyRepelle.register()
