import numpy as np
from tqdm import tqdm

from mushroom.algorithms.agent import Agent
from mushroom.utils.dataset import max_QA, parse_dataset
from mushroom.utils.replay_memory import Buffer


class BatchTD(Agent):
    """
    Implement functions to run batch algorithms.

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

        # "Boosted Fitted Q-Iteration". Tosatto S. et. al.. 2017.
        self._boosted = params['algorithm_params'].get('boosted', False)
        if self._boosted:
            self._old_prediction = 0.
            self._idx = 0

    def fit(self, dataset, n_iterations, target=None):
        """
        Fit loop.

        Args:
            dataset (list): the dataset;
            n_iterations (int): number of FQI iterations.
            target (np.array, None): initial target of FQI.

        Returns:
            Last target computed.

        """
        for _ in tqdm(xrange(n_iterations), dynamic_ncols=True,
                      disable=self._quiet, leave=False):
            target = self._partial_fit(dataset, target,
                                       **self.params['fit_params'])

        return target

    def _partial_fit(self, x, y, **fit_params):
        """
        Single fit iteration.

        Args:
            x (list): a two elements list with states and actions;
            y (np.array): targets;
            **fit_params (dict): other parameters to fit the model.

        Returns:
            Last target computed.

        """
        state, action, reward, next_state, absorbing, _ = parse_dataset(x)
        if y is None:
            if self._boosted:
                assert self._old_prediction == 0.
            target = reward
        else:
            maxq, _ = max_QA(next_state, absorbing, self.approximator)
            target = reward + self._gamma * maxq

        sa = [state, action]
        if self._boosted:
            target = target - self._old_prediction
            self.approximator[self._idx].fit(sa, target, **fit_params)
            self._old_prediction += self.approximator[self._idx].predict(sa)

            self._idx += 1
        else:
            self.approximator.fit(sa, target, **fit_params)

        return target


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
    """
    Deep Fitted Q-Iteration algorithm. This algorithm is used to apply FQI in
    dimensionally large problems. To fit the approximator, for memory reasons,
    this implementation expects the feature of the states extracted from an
    autoencoder, not the raw states.

    "Autonomous reinforcement learning on raw visual input data in a real world
    application". Lange S. et. al.. 2012.
    """
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

    def draw_action(self, state, approximator=None):
        self._buffer.add(state)

        if self._episode_steps < self._no_op_actions:
            action = np.array([self._no_op_action_value])
            self.policy.update()
        else:
            extended_state = self._buffer.get()

            feature_state = self._extractor.predict(
                np.expand_dims(extended_state, axis=0))[0]

            action = super(DeepFQI, self).draw_action(feature_state,
                                                      approximator=approximator)

        self._episode_steps += 1

        return action

    def episode_start(self):
        self._no_op_actions = np.random.randint(
            self._buffer.size, self._max_no_op_actions + 1)
        self._episode_steps = 0
