import numpy as np
from mushroom.algorithms.agent import Agent
from mushroom.approximators import Regressor
from mushroom.approximators.parametric import LinearApproximator


class COPDAC_Q(Agent):
    def __init__(self, policy, mu, mdp_info,
                 alpha_theta, alpha_omega, alpha_v,
                 value_function_features=None,
                 policy_features=None):

        self._mu = mu
        self._psi = value_function_features

        self._alpha_theta = alpha_theta
        self._alpha_omega = alpha_omega
        self._alpha_v = alpha_v

        if self._psi is not None:
            input_shape = (self._psi.size,)
        else:
            input_shape = mdp_info.observation_space.shape

        self._V = Regressor(LinearApproximator,
                            input_shape=input_shape,
                            output_shape=(1,))

        self._A = Regressor(LinearApproximator,
                            input_shape=(self._mu.weights_size,),
                            output_shape=(1,))

        super().__init__(policy, mdp_info, policy_features)

    def fit(self, dataset):
        for step in dataset:
            s, a, r, ss, absorbing, _ = step

            s_phi = self.phi(s) if self.phi is not None else s
            s_psi = self._psi(s) if self._psi is not None else s
            ss_psi = self._psi(ss) if self._psi is not None else ss

            q_next = np.asscalar(self._V(ss_psi)) if not absorbing else 0

            grad_mu_s = np.atleast_2d(self._mu.diff(s_phi))
            omega = self._A.get_weights()

            delta = r + self.mdp_info.gamma * q_next - self._Q(s, a)
            delta_theta = self._alpha_theta(s, a) * \
                          omega.dot(grad_mu_s.T).dot(grad_mu_s)
            delta_omega = self._alpha_omega(s, a)*delta*self._nu(s, a)
            delta_v = self._alpha_v(s, a)*delta*s_psi

            theta_new = self._mu.get_weights() + delta_theta
            self._mu.set_weights(theta_new)

            omega_new = omega + delta_omega
            self._A.set_weights(omega_new)

            v_new = self._V.get_weights() + delta_v
            self._V.set_weights(v_new)

    def _Q(self, state, action):
        state_psi = self._psi(state) if self._psi is not None else state
        return np.asscalar(self._V(state_psi)) + \
               np.asscalar(self._A(self._nu(state, action)))

    def _nu(self, state, action):
        state_phi = self.phi(state) if self.phi is not None else state
        grad_mu = np.atleast_2d(self._mu.diff(state_phi))
        delta = action - self._mu(state_phi)

        return delta.dot(grad_mu)


