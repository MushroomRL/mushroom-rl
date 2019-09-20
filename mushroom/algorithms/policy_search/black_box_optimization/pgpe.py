import numpy as np

from mushroom.algorithms.policy_search.black_box_optimization import BlackBoxOptimization


class PGPE(BlackBoxOptimization):
    """
    Policy Gradient with Parameter Exploration algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """
    def __init__(self, distribution, policy, mdp_info, learning_rate,
                 features=None):
        """
        Constructor.

        Args:
            learning_rate (Parameter): the learning rate for the gradient step.

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
            baseline_num_list.append(J_i * diff_log_dist2)
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