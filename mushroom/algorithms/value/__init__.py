from .batch_td import BatchTD, FQI, DoubleFQI, LSPI
from .dqn import DQN, DoubleDQN, AveragedDQN, CategoricalDQN
from .td import TD, QLearning, DoubleQLearning, WeightedQLearning,\
    SpeedyQLearning, RLearning, RQLearning, SARSA, SARSALambdaDiscrete,\
    SARSALambdaContinuous, ExpectedSARSA, TrueOnlineSARSALambda


__all__ = ['BatchTD', 'FQI', 'DoubleFQI', 'LSPI', 'TD', 'DQN', 'DoubleDQN',
           'AveragedDQN', 'CategoricalDQN', 'QLearning', 'DoubleQLearning',
           'WeightedQLearning', 'SpeedyQLearning', 'RLearning', 'RQLearning',
           'SARSA', 'SARSALambdaDiscrete', 'SARSALambdaContinuous',
           'ExpectedSARSA', 'TrueOnlineSARSALambda']
