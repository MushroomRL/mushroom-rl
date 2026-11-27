import numpy as np

from mushroom_rl.algorithms.policy_search.policy_gradient import PolicyGradient


class eNAC(PolicyGradient):
    """
    Episodic Natural Actor Critic algorithm.
    "A Survey on Policy Search for Robotics",
    Deisenroth M. P. et al. 2013.

    """
    def __init__(self, mdp_info, policy, optimizer, critic_features=None):
        """
        Constructor.

        Args:
            critic_features (Features, None): features used by the critic.

        """
        super().__init__(mdp_info, policy, optimizer)
        self.phi_c = critic_features

        self._add_save_attr(phi_c='pickle')

    def _compute_gradient(self, dataset):
        J = dataset.compute_J(self.mdp_info.gamma)
        d_log_pi = self._d_log_pi(dataset)
        discount = self.mdp_info.gamma ** np.arange(d_log_pi.shape[1])

        if self.phi_c is None:
            critic_features = np.ones((len(J), 1))
        else:
            critic_features = self.phi_c(dataset.get_init_states())

        psi = np.concatenate((np.sum(discount[:, None] * d_log_pi, axis=1), critic_features), axis=1)

        return (np.linalg.pinv(psi) @ J)[:self.policy.weights_size]
