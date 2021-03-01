import numpy as np

from scipy.optimize import minimize

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.utils.parameters import to_parameter


class REPS(BlackBoxOptimization):
    """
    Episodic Relative Entropy Policy Search algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """
    def __init__(self, mdp_info, distribution, policy, eps, features=None):
        """
        Constructor.

        Args:
            eps ([float, Parameter]): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.

        """
        self._eps = to_parameter(eps)

        self._add_save_attr(_eps='mushroom')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        eta_start = np.ones(1)

        res = minimize(REPS._dual_function, eta_start,
                       jac=REPS._dual_function_diff,
                       bounds=((np.finfo(np.float32).eps, np.inf),),
                       args=(self._eps(), Jep, theta))

        eta_opt = res.x.item()

        Jep -= np.max(Jep)

        d = np.exp(Jep / eta_opt)

        self.distribution.mle(theta, d)

    @staticmethod
    def _dual_function(eta_array, *args):
        eta = eta_array.item()
        eps, Jep, theta = args

        max_J = np.max(Jep)

        r = Jep - max_J
        sum1 = np.mean(np.exp(r / eta))

        return eta * eps + eta * np.log(sum1) + max_J

    @staticmethod
    def _dual_function_diff(eta_array, *args):
        eta = eta_array.item()
        eps, Jep, theta = args

        max_J = np.max(Jep)

        r = Jep - max_J

        sum1 = np.mean(np.exp(r / eta))
        sum2 = np.mean(np.exp(r / eta) * r)

        gradient = eps + np.log(sum1) - sum2 / (eta * sum1)

        return np.array([gradient])
