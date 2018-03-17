import numpy as np

from mushroom.algorithms.agent import Agent
from mushroom.utils.dataset import compute_J


class BlackBoxOptimization(Agent):
    """
    Base class for black box optimization algorithms.
    This algorithms work on a distribution of policy parameters,
    and often they do not rely on stochastic and differentiable
    policies.

    """
    def __init__(self, distribution, policy, mdp_info, features=None):
        """
        Constructor.

        Args:
            distribution: the distribution of policy parameters
            policy: the policy to use

        """
        self.distribution = distribution
        self._theta_list = list()

        super().__init__(policy, mdp_info, features)

    def episode_start(self):
        theta = self.distribution.sample()
        self._theta_list.append(theta)
        self.policy.set_weights(theta)

    def fit(self, dataset):
        Jep = compute_J(dataset, self.mdp_info.gamma)

        Jep = np.array(Jep)
        theta = np.array(self._theta_list)

        self._update(Jep, theta)

        self._theta_list = list()

    def stop(self):
        self._theta_list = list()

    def _update(self, Jep, theta):
        """
        Function that implements the update routine of distribution parameters.
        Every black box algorithms should implement this function with the proper update.

        Args:
            Jep (np.ndarray): a vector containing the J of the considered trajectories
            theta (np.ndarray): a matrix of policy parameters of the considered trajectories

        """
        raise NotImplementedError('BlackBoxOptimization is an abstract class')


class RWR(BlackBoxOptimization):
    def __init__(self, distribution, policy, mdp_info, beta, features=None):
        """
        Constructor.

        Args:
            beta (float): the temperature for the exponential
             reward transformation.

        """
        self.beta = beta

        super().__init__(distribution, policy, mdp_info, features)

    def _update(self, Jep, theta):
        Jep -= np.max(Jep)

        d = np.exp(self.beta * Jep)
        theta = np.array(theta)

        self.distribution.mle(theta, d)


class PGPE(BlackBoxOptimization):
    def __init__(self, distribution, policy, mdp_info, learning_rate, features=None):
        """
        Constructor.

        Args:
            learning_rate (Parameter): the learning rate for the gradient step

        """
        self.learning_rate = learning_rate

        super().__init__(distribution, policy, mdp_info, features)

    def _update(self, Jep, theta):
        baseline_num_list = list()
        baseline_den_list = list()
        diff_log_dist_list = list()

        # Compute derivatives of distribution and baseline components
        for i in range(len(Jep)):
            J_i = Jep[i]
            theta_i = theta[i]

            diff_log_dist = self.distribution.diff_log(theta_i)
            diff_log_dist2 = diff_log_dist**2

            diff_log_dist_list.append(diff_log_dist)
            baseline_num_list.append(J_i*diff_log_dist2)
            baseline_den_list.append(diff_log_dist2)

        # Compute baseline
        baseline = np.mean(baseline_num_list, axis=0) / \
                   np.mean(baseline_den_list, axis=0)
        baseline[np.logical_not(np.isfinite(baseline))] = 0.

        # Compute gradient
        grad_J_list = list()
        for i in range(len(Jep)):
            diff_log_dist = diff_log_dist_list[i]
            J_i = Jep[i]

            grad_J_list.append(diff_log_dist * (J_i - baseline))

        grad_J = np.mean(grad_J_list, axis=0)

        omega = self.distribution.get_parameters()
        omega += self.learning_rate(grad_J) * grad_J
        self.distribution.set_parameters(omega)
