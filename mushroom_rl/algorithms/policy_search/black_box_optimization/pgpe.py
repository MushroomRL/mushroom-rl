import numpy as np

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization


class PGPE(BlackBoxOptimization):
    """
    Policy Gradient with Parameter Exploration algorithm.
    "A Survey on Policy Search for Robotics",
    Deisenroth M. P. et al. 2013.

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
        J = Jep.reshape(-1, 1)
        diff_log_dist = np.array([self.distribution.diff_log(theta_i, context) for theta_i in theta])

        divisor = np.sum(diff_log_dist**2, axis=0)
        zero_divisor = divisor == 0.
        divisor[zero_divisor] = 1.
        baseline = np.sum(J * diff_log_dist**2, axis=0) / divisor
        baseline[zero_divisor] = 0.

        grad_J = np.mean(diff_log_dist * (J - baseline), axis=0)

        omega_old = self.distribution.get_parameters()
        omega_new = self.optimizer(omega_old, grad_J)
        self.distribution.set_parameters(omega_new)
