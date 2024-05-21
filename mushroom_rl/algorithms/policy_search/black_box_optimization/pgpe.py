import numpy as np

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization


class PGPE(BlackBoxOptimization):
    """
    Policy Gradient with Parameter Exploration algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """
    def __init__(self, mdp_info, distribution, policy, optimizer, context_builder=None):
        """
        Constructor.

        Args:
            optimizer: the gradient step optimizer.

        """
        self.optimizer = optimizer

        super().__init__(mdp_info, distribution, policy, context_builder=context_builder)

        self._add_save_attr(optimizer='mushroom')

    def _update(self, Jep, theta, context):
        baseline_num_list = list()
        baseline_den_list = list()
        diff_log_dist_list = list()

        # Compute derivatives of distribution and baseline components
        for i in range(len(Jep)):
            J_i = Jep[i]
            theta_i = theta[i]

            diff_log_dist = self.distribution.diff_log(theta_i, context)
            diff_log_dist2 = diff_log_dist**2

            diff_log_dist_list.append(diff_log_dist)
            baseline_num_list.append(J_i * diff_log_dist2)
            baseline_den_list.append(diff_log_dist2)

        # Compute baseline
        baseline = np.mean(baseline_num_list, axis=0) / np.mean(baseline_den_list, axis=0)
        baseline[np.logical_not(np.isfinite(baseline))] = 0.

        # Compute gradient
        grad_J_list = list()
        for i in range(len(Jep)):
            diff_log_dist = diff_log_dist_list[i]
            J_i = Jep[i]

            grad_J_list.append(diff_log_dist * (J_i - baseline))

        grad_J = np.mean(grad_J_list, axis=0)

        omega_old = self.distribution.get_parameters()
        omega_new = self.optimizer(omega_old, grad_J)
        self.distribution.set_parameters(omega_new)
