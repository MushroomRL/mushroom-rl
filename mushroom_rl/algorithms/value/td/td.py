import numpy as np

from mushroom_rl.core import Agent


class TD(Agent):
    """
    Implements functions to run TD algorithms.

    """
    def __init__(self, mdp_info, policy, approximator, learning_rate,
                 features=None):
        """
        Constructor.

        Args:
            approximator (object): the approximator to use to fit the
               Q-function;
            learning_rate (Parameter): the learning rate.

        """
        self._alpha = learning_rate

        policy.set_q(approximator)
        self.Q = approximator

        self._add_save_attr(_alpha='mushroom', Q='mushroom')

        super().__init__(mdp_info, policy, features)

    def fit(self, dataset, **info):
        assert len(dataset) == 1

        state, action, reward, next_state, absorbing = self._parse(dataset)
        self._update(state, action, reward, next_state, absorbing)

    @staticmethod
    def _parse(dataset):
        """
        Utility to parse the dataset that is supposed to contain only a sample.

        Args:
            dataset (list): the current episode step.

        Returns:
            A tuple containing state, action, reward, next state, absorbing and
            last flag.

        """
        sample = dataset[0]
        state = sample[0]
        action = sample[1]
        reward = sample[2]
        next_state = sample[3]
        absorbing = sample[4]

        return state, action, reward, next_state, absorbing

    def _update(self, state, action, reward, next_state, absorbing):
        """
        Update the Q-table.

        Args:
            state (np.ndarray): state;
            action (np.ndarray): action;
            reward (np.ndarray): reward;
            next_state (np.ndarray): next state;
            absorbing (np.ndarray): absorbing flag.

        """
        pass

    def _post_load(self):
        self.policy.set_q(self.Q)