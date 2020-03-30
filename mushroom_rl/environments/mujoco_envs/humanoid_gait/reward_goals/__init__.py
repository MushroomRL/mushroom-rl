from .Reward import NoGoalReward, MaxVelocityReward, \
    VelocityProfileReward, CompleteTrajectoryReward

from .Trajectory import CompleteHumanoidTrajectory, HumanoidTrajectory

from .VelocityProfile import VelocityProfile, PeriodicVelocityProfile, SinVelocityProfile, ConstantVelocityProfile, \
    RandomConstantVelocityProfile, SquareWaveVelocityProfile,  VelocityProfile3D