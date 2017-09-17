import numpy as np
from tqdm import tqdm

from mushroom.algorithms.agent import Agent
from mushroom.utils.dataset import max_QA, parse_dataset
from mushroom.utils.replay_memory import Buffer


class BatchTD(Agent):
    """
    Implements functions to run batch algorithms.

    """
    def __init__(self, approximator, policy, gamma, **params):
        self._quiet = params.get('quiet', False)
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
        for _ in tqdm(xrange(n_iterations), dynamic_ncols=True,
                      disable=self._quiet, leave=False):
            target = self._partial_fit(dataset, target,
                                       **self.params['fit_params'])

    def _partial_fit(self, x, y, **fit_params):
        """
        Single fit iteration.

        Args:
            x (list): a two elements list with states and actions;
            y (np.array): targets;
            **fit_params (dict): other parameters to fit the model.

        """
        state, action, reward, next_state, absorbing, _ = parse_dataset(x)
        if y is None:
            y = reward
        else:
            maxq, _ = max_QA(next_state, absorbing, self.approximator)
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

    def _partial_fit(self, x, y, **fit_params):
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

    def _partial_fit(self, x, y, **fit_params):
        pass


class DeepFQI(FQI):
    def __init__(self, approximator, policy, gamma, **params):
        self.__name__ = 'DeepFQI'

        alg_params = params['algorithm_params']
        self._buffer = Buffer(size=alg_params.get('history_length', 1))
        self._extractor = alg_params.get('extractor')
        self._max_no_op_actions = alg_params.get('max_no_op_actions')
        self._no_op_action_value = alg_params.get('no_op_action_value')
        self._episode_steps = 0
        self._no_op_actions = None

        super(DeepFQI, self).__init__(approximator, policy, gamma, **params)

    def _partial_fit(self, x, y, **fit_params):
        """
        Single fit iteration.

        Args:
            x (list): a two elements list with states and actions;
            y (np.array): targets;
            **fit_params (dict): other parameters to fit the model.

        """
        state, action, reward, next_state, absorbing, _ = x
        if y is None:
            y = reward
        else:
            q = np.ones((next_state.shape[1], next_state.shape[0]))
            for i in xrange(q.shape[1]):
                sa_n = [next_state[i], np.ones((next_state[i].shape[0], 1)) * i]
                q[:, i] = self.approximator.predict(sa_n)
            if np.any(absorbing):
                q *= 1 - absorbing.reshape(-1, 1)
            maxq = np.max(q, axis=1)
            y = reward + self._gamma * maxq

        sa = [state, action]
        self.approximator.fit(sa, y, **fit_params)

        return y

    def draw_action(self, state, approximator=None):
        self._buffer.add(state)

        if self._episode_steps < self._no_op_actions:
            action = np.array([self._no_op_action_value])
        else:
            extended_state = self._buffer.get()

            if not np.random.uniform() < self.policy._epsilon(extended_state):
                q = np.ones(self.mdp_info['action_space'].n)
                for i in xrange(q.size):
                    sa = [np.expand_dims(extended_state, axis=0),
                          np.ones((1, 1)) * i]
                    features = self._extractor.predict(sa)
                    fa = [features, np.ones((1, 1)) * i]
                    q[i] = self.approximator.predict(fa)
                action = np.array(
                    [np.random.choice(np.argwhere(q == np.max(q)).ravel())])
            else:
                action = self.mdp_info['action_space'].sample()

        self._episode_steps += 1
        self.policy.update()

        return action

    def episode_start(self):
        self._no_op_actions = np.random.randint(
            self._buffer.size, self._max_no_op_actions + 1)
        self._episode_steps = 0
