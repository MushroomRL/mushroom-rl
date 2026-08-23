from .td import (TD, SARSA, SARSALambda, ExpectedSARSA, QLearning, QLambda, DoubleQLearning, SpeedyQLearning,
                 RLearning, WeightedQLearning, MaxminQLearning, RQLearning, RQLearningOnPolicy, SARSALambdaContinuous,
                 TrueOnlineSARSALambda)
from .batch_td import BatchTD, FQI, DoubleFQI, BoostedFQI, LSPI
from .dqn import (AbstractDQN, DQN, DoubleDQN, AveragedDQN, CategoricalDQN, DuelingDQN, NoisyDQN, QuantileDQN,
                  MaxminDQN, Rainbow)


__all__ = ['TD', 'QLearning', 'QLambda', 'DoubleQLearning', 'WeightedQLearning', 'MaxminQLearning', 'SpeedyQLearning',
           'RLearning', 'RQLearning', 'RQLearningOnPolicy', 'SARSA', 'SARSALambda', 'SARSALambdaContinuous',
           'ExpectedSARSA', 'TrueOnlineSARSALambda', 'BatchTD', 'FQI', 'DoubleFQI', 'BoostedFQI', 'LSPI',
           'AbstractDQN', 'DQN', 'DoubleDQN', 'AveragedDQN', 'CategoricalDQN', 'DuelingDQN', 'NoisyDQN', 'QuantileDQN',
           'MaxminDQN', 'Rainbow', ]
