import numpy as np

from mushroom_rl.core import Agent


class PolicyGradient(Agent):
    """
    Abstract class to implement a generic Policy Search algorithm using the
    gradient of the policy to update its parameters.
    "A survey on Policy Gradient algorithms for Robotics". Deisenroth M. P. et
    al.. 2011.

    """
    def __init__(self, mdp_info, policy, optimizer):
        """
        Constructor.

        Args:
            optimizer: the gradient optimizer.

        """
        self.optimizer = optimizer
        self.df = 1
        self.J_episode = 0

        self._add_save_attr(
            optimizer='mushroom',
            df='numpy',
            J_episode='numpy'
        )

        super().__init__(mdp_info, policy)

    def fit(self, dataset):
        J = list()
        self.df = 1.
        self.J_episode = 0.
        self._init_update()
        for sample in dataset:
            x, u, r, xn, _, last = self._parse(sample)
            self._step_update(x, u, r)
            self.J_episode += self.df * r
            self.df *= self.mdp_info.gamma

            if last:
                self._episode_end_update()
                J.append(self.J_episode)
                self.J_episode = 0.
                self.df = 1.
                self._init_update()
        
        assert len(J) > 1, "More than one episode is needed to compute the gradient"
        
        self._update_parameters(J)

    def _update_parameters(self, J):
        """
        Update the parameters of the policy.

        Args:
             J (list): list of the cumulative discounted rewards for each
                episode in the dataset.

        """
        grad = self._compute_gradient(J)

        theta_old = self.policy.get_weights()

        theta_new = self.optimizer(theta_old, grad)

        self.policy.set_weights(theta_new)

    def _init_update(self):
        """
        This function is called, when parsing the dataset, at the beginning
        of each episode. The implementation is dependent on the algorithm (e.g.
        REINFORCE resets some data structure).

        """
        raise NotImplementedError('PolicyGradient is an abstract class')

    def _step_update(self, x, u, r):
        """
        This function is called, when parsing the dataset, at each episode step.

        Args:
            x (np.ndarray): the state at the current step;
            u (np.ndarray): the action at the current step;
            r (np.ndarray): the reward at the current step.

        """
        raise NotImplementedError('PolicyGradient is an abstract class')

    def _episode_end_update(self):
        """
        This function is called, when parsing the dataset, at the beginning
        of each episode. The implementation is dependent on the algorithm (e.g.
        REINFORCE updates some data structures).

        """
        raise NotImplementedError('PolicyGradient is an abstract class')

    def _compute_gradient(self, J):
        """
        Return the gradient computed by the algorithm.

        Args:
             J (list): list of the cumulative discounted rewards for each
                episode in the dataset.

        Returns:
            The gradient computed by the algorithm.

        """
        raise NotImplementedError('PolicyGradient is an abstract class')

    def _parse(self, sample):
        """
        Utility to parse the sample.

        Args:
             sample (list): the current episode step.

        Returns:
            A tuple containing state, action, reward, next state, absorbing and
            last flag. If provided, ``state`` is preprocessed with the features.

        """
        state = sample[0]
        action = sample[1]
        reward = sample[2]
        next_state = sample[3]
        absorbing = sample[4]
        last = sample[5]

        return state, action, reward, next_state, absorbing, last
