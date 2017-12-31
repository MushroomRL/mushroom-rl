from .batch_td import FQI, DoubleFQI, LSPI
from .dqn import DQN, DoubleDQN, AveragedDQN
from .td import QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning,\
    RLearning, RQLearning, SARSA, SARSALambdaDiscrete, SARSALambdaContinuous,\
    ExpectedSARSA, TrueOnlineSARSALambda


__all__ = ['FQI', 'DoubleFQI', 'LSPI', 'DQN', 'DoubleDQN',
           'AveragedDQN', 'QLearning', 'DoubleQLearning',
           'WeightedQLearning', 'SpeedyQLearning', 'RLearning', 'RQLearning',
           'SARSA', 'SARSALambdaDiscrete', 'SARSALambdaContinuous',
           'ExpectedSARSA', 'TrueOnlineSARSALambda']
