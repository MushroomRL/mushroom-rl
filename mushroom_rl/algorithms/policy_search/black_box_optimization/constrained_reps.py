import numpy as np
from scipy.optimize import minimize
from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.utils.parameters import to_parameter


class ConstrainedREPS(BlackBoxOptimization):
    """
    Episodic Relative Entropy Policy Search algorithm with constrained policy update.

    """
    def __init__(self, mdp_info, distribution, policy, eps, kappa, features=None):
        """
        Constructor.

        Args:
            eps ([float, Parameter]): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.
            kappa ([float, Parameter]): the maximum admissible value for the entropy decrease
                between the new distribution and the 
                previous one at each update step. 

        """
        self._eps = to_parameter(eps)
        self._kappa = to_parameter(kappa)

        self._add_save_attr(_eps='mushroom')
        self._add_save_attr(_kappa='mushroom')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        eta_start = np.ones(1)

        res = minimize(ConstrainedREPS._dual_function, eta_start,
                       jac=ConstrainedREPS._dual_function_diff,
                       bounds=((np.finfo(np.float32).eps, np.inf),),
                       args=(self._eps(), Jep, theta))

        eta_opt = res.x.item()

        Jep -= np.max(Jep)

        d = np.exp(Jep / eta_opt)

        self.distribution.con_wmle(theta, d, self._eps(), self._kappa())

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
