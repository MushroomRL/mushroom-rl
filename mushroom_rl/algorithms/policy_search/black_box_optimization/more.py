import numpy as np
from scipy.optimize import minimize

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.features import Features
from mushroom_rl.features.basis.polynomial import PolynomialBasis
from mushroom_rl.distributions import GaussianCholeskyDistribution


class MORE(BlackBoxOptimization):
    """
    Model-Based Relative Entropy Stochastic Search algorithm.
    "Model-Based Relative Entropy Stochastic Search", Abdolmaleki, Abbas and Lioutikov, Rudolf and
    Peters, Jan R and Lau, Nuno and Pualo Reis, Luis and Neumann, Gerhard. 2015.

    """
    def __init__(self, mdp_info, distribution, policy, eps, h0=-75, kappa=0.99, features=None):
        """
        Constructor.

        Args:
            distribution (GaussianCholeskyDistribution): the distribution of policy parameters.
            eps ([float, Parameter]): the maximum admissible value for the Kullback-Leibler
                divergence between the new distribution and the
                previous one at each update step.
            h0 ([float, Parameter]): minimum exploration policy.
            kappa ([float, Parameter]): regularization parameter for the entropy decrease.

        """

        self.eps = to_parameter(eps)
        self.h0 = to_parameter(h0)
        self.kappa = to_parameter(kappa)

        assert isinstance(distribution, GaussianCholeskyDistribution)

        poly_basis_quadratic = PolynomialBasis().generate(2, policy.weights_size)
        self.phi_quadratic_ = Features(basis_list=poly_basis_quadratic)
        self.regressor_quadratic = Regressor(LinearApproximator,
                      input_shape=(len(poly_basis_quadratic),),
                      output_shape=(1,))

        poly_basis_linear = PolynomialBasis().generate(1, policy.weights_size)
        self.phi_linear_ = Features(basis_list=poly_basis_linear)
        self.regressor_linear = Regressor(LinearApproximator,
                      input_shape=(len(poly_basis_linear),),
                      output_shape=(1,))
        
        self._add_save_attr(eps='primitive')
        self._add_save_attr(h0='primitive')
        self._add_save_attr(kappa='primitive')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        
        beta = self.kappa() * (self.distribution.entropy() - self.h0()) + self.h0()

        n = len(self.distribution._mu)
        dist_params = self.distribution.get_parameters()
        mu_t = dist_params[:n][:,np.newaxis]
        chol_sig_empty = np.zeros((n,n))
        chol_sig_empty[np.tril_indices(n)] = dist_params[n:]
        sig_t = chol_sig_empty.dot(chol_sig_empty.T)

        R, r, r_0 = self._fit_quadratic_surrogate(theta, Jep, n)
        
        eta_omg_start = np.ones(2)
        res = minimize(MORE._dual_function, eta_omg_start,
                       bounds=((np.finfo(np.float32).eps, np.inf),(np.finfo(np.float32).eps, np.inf)),
                       args=(sig_t, mu_t, R, r, r_0, self.eps(), beta, n),
                       method=None)

        eta_opt, omg_opt = res.x[0], res.x[1]
        
        mu_t1, sig_t1 = MORE._closed_form_mu_t1_sig_t1(sig_t, mu_t, R, r, eta_opt, omg_opt)

        dist_params = np.concatenate((mu_t1.flatten(), np.linalg.cholesky(sig_t1)[np.tril_indices(n)].flatten()))
        self.distribution.set_parameters(dist_params)
    
    def _fit_quadratic_surrogate(self, theta, Jep, n):

        Jep = ( Jep - np.mean(Jep, keepdims=True, axis=0) ) / np.std(Jep, keepdims=True, axis=0)
        
        features_quadratic = self.phi_quadratic_(theta)
        self.regressor_quadratic.fit(features_quadratic, Jep)        
        beta = self.regressor_quadratic.get_weights()

        R = np.zeros((n,n ))
        uid = np.triu_indices(n)
        R[uid] = beta[1 + n:]
        R.T[uid] = R[uid]
        
        w, v = np.linalg.eig(R)
        w[w >= 0.0] = -1e-12
        R = v @ np.diag(w) @ v.T
        R = 0.5 * (R + R.T)
        
        features_linear = self.phi_linear_(theta)
        aux = Jep - np.einsum('nk,kh,nh->n', theta, R, theta)
        self.regressor_linear.fit(features_linear, aux)
        beta = self.regressor_linear.get_weights()

        r_0 = beta[0]
        r = beta[1:][:,np.newaxis]

        return R, r, r_0
    
    @staticmethod
    def _dual_function(lag_array, Q, b, R, r, r_0, eps, kappa, n):
        eta, omg = lag_array[0], lag_array[1]
        F, f = MORE._get_F_f(Q, b, R, r, eta)
        slogdet_0 = np.linalg.slogdet( (2*np.pi) * Q )
        slogdet_1 = np.linalg.slogdet( (2*np.pi) * (eta + omg) * F )
        term1 = (f.T @ F @ f) - eta * (b.T @ np.linalg.inv(Q) @ b) - eta * slogdet_0[1] + (eta + omg) * slogdet_1[1] + r_0
        
        return eta * eps - omg * kappa + 0.5 * term1[0]

    @staticmethod
    def _closed_form_mu_t1_sig_t1(Q, b, R, r, eta, omg):
        F, f = MORE._get_F_f(Q, b, R, r, eta)
        mu_t1 = F @ f
        sig_t1 = F * (eta + omg)
        
        return mu_t1, sig_t1

    @staticmethod
    def _get_F_f(Q, b, R, r, eta):
        Q_inv = np.linalg.inv(Q)
        F = np.linalg.inv(eta * Q_inv - 2. * R)
        f = eta * Q_inv @ b + r
        
        return F, f
  
