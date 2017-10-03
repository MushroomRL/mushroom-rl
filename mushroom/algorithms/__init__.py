from .batch_td import BatchTD, FQI, DoubleFQI, WeightedFQI, DeepFQI
from .dqn import DQN, DoubleDQN, WeightedDQN
from .td import TD, QLearning, DoubleQLearning, WeightedQLearning, SARSA

__all__ = ['Algorithm', 'BatchTD', 'FQI', 'DeepFQI', 'DQN', 'DoubleDQN',
           'WeightedDQN', 'DoubleFQI', 'WeightedFQI', 'TD', 'QLearning',
           'DoubleQLearning', 'WeightedQLearning', 'SARSA']
