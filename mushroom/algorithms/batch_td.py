import numpy as np

from mushroom.algorithms.agent import Agent
from mushroom.utils.dataset import max_QA, parse_dataset
from mushroom.utils.replay_memory import Buffer, ReplayMemory


class BatchTD(Agent):
    """
    Implements functions to run batch algorithms.

    """
    def __init__(self, approximator, policy, gamma, **params):
        super(BatchTD, self).__init__(approximator, policy, gamma, **params)

    def __str__(self):
        return self.__name__


class FQI(BatchTD):
    """
    Fitted Q-Iteration algorithm.
    "Tree-Based Batch Mode Reinforcement Learning", Ernst D. et.al.. 2005.

    """
    def __init__(self, approximator, policy, gamma, **params):
        self.__name__ = 'FQI'

        super(FQI, self).__init__(approximator, policy, gamma, **params)

    def fit(self, dataset, n_iterations):
        """
        Fit loop.

        Args:
            dataset (list): the dataset;
            n_iterations (int): number of FQI iterations.

        """
        target = None
        for i in xrange(n_iterations):
            target = self.partial_fit(dataset, target,
                                      **self.params['fit_params'])

    def partial_fit(self, x, y, **fit_params):
        """
        Single fit iteration.

        Args:
            x (list): a two elements list with states and actions;
            y (np.array): targets;
            **fit_params (dict): other parameters to fit the model.

        """
        state, action, reward, next_states, absorbing, last = parse_dataset(x)
        if y is None:
            y = reward
        else:
            maxq, _ = max_QA(next_states, absorbing, self.approximator)
            y = reward + self._gamma * maxq

        sa = [state, action]
        self.approximator.fit(sa, y, **fit_params)

        return y


class DoubleFQI(FQI):
    """
    Double Fitted Q-Iteration algorithm.
    "Estimating the Maximum Expected Value in Continuous Reinforcement Learning
    Problems". D'Eramo C. et. al.. 2017.

    """
    def __init__(self, approximator, policy, gamma, **params):
        self.__name__ = 'DoubleFQI'

        super(DoubleFQI, self).__init__(approximator, policy, gamma, **params)

    def partial_fit(self, x, y, **fit_params):
        pass


class WeightedFQI(FQI):
    """
    Weighted Fitted Q-Iteration algorithm.
    "Estimating the Maximum Expected Value in Continuous Reinforcement Learning
    Problems". D'Eramo C. et. al.. 2017.

    """
    def __init__(self, approximator, policy, gamma, **params):
        self.__name__ = 'WeightedFQI'

        super(WeightedFQI, self).__init__(approximator, policy, gamma, **params)

    def partial_fit(self, x, y, **fit_params):
        pass


class DeepFQI(FQI):
    """
    Deep Fitted Q-Iteration algorithm.

    "Deep Auto-Encoder Neural Networks in Reinforcement Learning". Lange S. and
    Riedmiller M.. 2010.

    """
    def __init__(self, approximator, policy, gamma, **params):
        self.__name__ = 'DeepFQI'

        alg_params = params['algorithm_params']
        self._extractor = alg_params.get('extractor')
        self._batch_size = alg_params.get('batch_size')
        self._clip_reward = alg_params.get('clip_reward', True)
        self._replay_memory = ReplayMemory(alg_params.get('dataset_size'),
                                           alg_params.get('history_length', 1))
        self._buffer = Buffer(size=alg_params.get('history_length', 1))
        self._max_no_op_actions = alg_params.get('max_no_op_actions', 0)
        self._no_op_action_value = alg_params.get('no_op_action_value', 0)

        self._episode_steps = None
        self._no_op_actions = None

        super(DeepFQI, self).__init__(approximator, policy, gamma, **params)

    def fit(self, dataset, n_iterations):
        self._replay_memory.add(dataset)
        state, action, reward, next_state, absorbing, last = \
            self._replay_memory.get(self._replay_memory.size)

        sa = [state, action]
        self._extractor.fit(sa, next_state, **self.params['fit_params'])

        sa_n = [next_state, action]
        feature_dataset = [self._extractor.predict(sa), action, reward,
                           self._extractor.predict(sa_n), absorbing, last]

        super(DeepFQI, self).fit(feature_dataset, n_iterations)

    def initialize(self, mdp_info):
        """
        Initialize mdp info attribute.

        Args:
            mdp_info (dict): information about the mdp (e.g. discount factor).

        """
        super(DeepFQI, self).initialize(mdp_info)

        self._replay_memory.initialize(self.mdp_info)

    def draw_action(self, state):
        self._buffer.add(state)

        if self._episode_steps < self._no_op_actions:
            action = np.array([self._no_op_action_value])
            self.policy.update()
        else:
            extended_state = self._buffer.get()

            action = super(DeepFQI, self).draw_action(extended_state)

        self._episode_steps += 1

        return action

    def episode_start(self):
        self._no_op_actions = np.random.randint(
            self._replay_memory._history_length, self._max_no_op_actions + 1)
        self._episode_steps = 0

    def __str__(self):
        return self.__name__
