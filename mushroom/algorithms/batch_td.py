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
    Deep Fitted Q-Iteration algorithm.

    "Deep Auto-Encoder Neural Networks in Reinforcement Learning". Lange S. and
    Riedmiller M.. 2010.

    """
    def __init__(self, approximator, policy, gamma, **params):
        self.__name__ = 'DeepFQI'

        alg_params = params['algorithm_params']
        self._extractor = alg_params.get('extractor')
        self._n_epochs = alg_params.get('n_epochs')
        self._predict_next_state = alg_params.get('predict_next_state', False)
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
        replay_memory_generator = self._replay_memory.generator(
            self._batch_size
        )

        if self._predict_next_state:
            assert hasattr(self._extractor, 'discrete_actions')

            for e in xrange(self._n_epochs):
                for batch in replay_memory_generator:
                    sa = [batch[0], batch[1]]
                    self._extractor.train_on_batch(sa,
                                                   batch[3],
                                                   **self.params['fit_params'])

            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(len(dataset))
            sa = [state, action]
            state_feature = self._extractor.predict(sa)
            next_state_feature = np.ones(
                (np.unique(action).size,) + state_feature.shape)
            for i in xrange(next_state_feature.shape[0]):
                a = np.ones((state_feature.shape[0], 1)) * i
                sa_n = [next_state, a]
                next_state_feature[i, :] = self._extractor.predict(sa_n)

            feature_dataset = [state_feature, action, reward,
                               next_state_feature,
                               absorbing]

            target = None
            for i in xrange(n_iterations):
                target = self._partial_fit_next_state(
                    feature_dataset,
                    target,
                    **self.params['fit_params']
                )
        else:
            assert not hasattr(self._extractor, 'discrete_actions')

            for e in xrange(self._n_epochs):
                for batch in replay_memory_generator:
                    self._extractor.train_on_batch(batch[0],
                                                   batch[0],
                                                   **self.params['fit_params'])

            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._replay_memory.size)

            self._extractor.train_on_batch(
                state, state, **self.params['fit_params'])
            state_feature = self._extractor.predict(state)
            next_state_feature = self._extractor.predict(next_state)

            feature_dataset = [state_feature, action, reward,
                               next_state_feature,
                               absorbing]

            target = None
            for i in xrange(n_iterations):
                target = self._partial_fit_state(feature_dataset, target,
                                                 **self.params['fit_params'])

    def _partial_fit_next_state(self, x, y, **fit_params):
        assert not hasattr(self.approximator, 'discrete_actions')

        if y is None:
            y = x[2]
        else:
            q = np.ones((x[0].shape[0], np.unique(x[1]).size))
            for i in xrange(q.shape[1]):
                q[:, i] = self.approximator.predict(x[3][i])

            if np.any(x[4]):
                q *= 1 - x[4].reshape(-1, 1)
            max_q = np.max(q, axis=1)
            y = x[2] + self._gamma * max_q

        self.approximator.fit(x[0], y, **fit_params)

        return y

    def _partial_fit_state(self, x, y, **fit_params):
        assert hasattr(self.approximator, 'discrete_actions')

        if y is None:
            y = x[2]
        else:
            q = self.approximator.predict_all(x[0])

            if np.any(x[4]):
                q *= 1 - x[4].reshape(-1, 1)
            max_q = np.max(q, axis=1)
            y = x[2] + self._gamma * max_q

        sa = [x[0], x[1]]

        self.approximator.fit(sa, y, **fit_params)

        return y

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
            return self.policy(np.expand_dims(state, axis=0), self.approximator)

            extended_state = self._buffer.get()

            n_actions = self._extractor._discrete_actions.shape[0]
            if self._predict_next_state:
                for i in xrange(n_actions):
                    a = np.ones((extended_state.shape[0], 1)) * i
                    sa = [np.array([extended_state]), a]
                    feature = self._extractor.predict(sa)
            else:
                feature = self._extractor.predict(np.array([extended_state]))

            action = super(DeepFQI, self).draw_action(feature)

        self._episode_steps += 1

        return action

    def episode_start(self):
        self._no_op_actions = np.random.randint(
            self._replay_memory._history_length, self._max_no_op_actions + 1)
        self._episode_steps = 0

    def __str__(self):
        return self.__name__
