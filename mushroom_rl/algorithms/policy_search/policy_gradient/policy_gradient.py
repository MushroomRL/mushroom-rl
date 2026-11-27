import numpy as np

from mushroom_rl.core import Agent
from mushroom_rl.utils.episodes import split_episodes


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

        self._add_save_attr(optimizer='mushroom')

        super().__init__(mdp_info, policy)

    def fit(self, dataset):
        J = dataset.compute_J(self.mdp_info.gamma)
        assert len(J) > 1, "More than one episode is needed to compute the gradient"

        grad = self._compute_gradient(dataset)
        self.policy.set_weights(self.optimizer(self.policy.get_weights(), grad))

        if self._logger:
            self._logger.log_training(J=np.mean(J).item())
            self._logger.advance_step()

    def _compute_gradient(self, dataset):
        """
        Compute the gradient of the expected return w.r.t. the policy weights.

        Args:
            dataset (Dataset): the complete episodes the gradient is estimated from.

        Returns:
            The gradient, of shape ``(n_weights,)``.

        """
        raise NotImplementedError('PolicyGradient is an abstract class')

    def _d_log_pi(self, dataset):
        """
        Compute the gradient of the log-probability of each action in the dataset w.r.t. the policy weights.

        Args:
            dataset (Dataset): the complete episodes.

        Returns:
            The gradients, grouped by episode and padded with zeros to the length ``T`` of the longest one, of shape
            ``(n_episodes, T, n_weights)``.

        """
        d_log_pi = np.array([self.policy.diff_log(x, u) for x, u in zip(dataset.state, dataset.action)])

        return split_episodes(dataset.last, d_log_pi)[1]
