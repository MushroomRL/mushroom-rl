from .batch_td import FQI, DoubleFQI, WeightedFQI, LSPI
from .dqn import DQN, DoubleDQN, AveragedDQN
from .td import QLearning, DoubleQLearning, WeightedQLearning, SpeedyQLearning,\
    RLearning, RQLearning, SARSA, SARSALambdaDiscrete, SARSALambdaContinuous,\
    ExpectedSARSA, TrueOnlineSARSALambda


__all__ = ['FQI', 'DoubleFQI', 'WeightedFQI', 'LSPI', 'DQN', 'DoubleDQN',
           'AveragedDQN', 'QLearning', 'DoubleQLearning',
           'WeightedQLearning', 'SpeedyQLearning', 'RLearning', 'RQLearning',
           'SARSA', 'SARSALambdaDiscrete', 'SARSALambdaContinuous',
           'ExpectedSARSA', 'TrueOnlineSARSALambda']
