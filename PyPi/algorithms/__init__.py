from .batch_td import BatchTD, FQI, DoubleFQI, WeightedFQI
from .dqn import DQN, DoubleDQN
from .td import TD, QLearning, DoubleQLearning, WeightedQLearning, SARSA

__all__ = ['Algorithm', 'BatchTD', 'FQI', 'DQN', 'DoubleDQN', 'DoubleFQI',
           'WeightedFQI', 'TD', 'QLearning', 'DoubleQLearning',
           'WeightedQLearning', 'SARSA']
