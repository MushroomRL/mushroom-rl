from copy import deepcopy

import numpy as np

from mushroom.algorithms import Agent
from mushroom.approximators import Regressor


class PPO(Agent):
    """
    Proximal Policy Optimization algorithm.
    "Proximal Policy Optimization Algorithms". Schulman J. et al.. 2017.

    """
    def __init__(self, policy, mu, mdp_info, epsilon, n_actors,
                 value_function_approximator, clipped=True, beta=0., d_targ=0.,
                 c1=0., c2=0., lambda_par=1., value_function_features=None,
                 policy_features=None):
        """
        Constructor.

        Args:
             ...

        """
        self._mu = mu
        self._psi = value_function_features

        self._n_actors = n_actors
        self._epsilon = epsilon
        self._clipped = clipped
        self._beta = beta
        self._d_targ = d_targ
        self._c1 = c1
        self._c2 = c2
        self._lambda = lambda_par

        if self._psi is not None:
            input_shape = (self._psi.size,)
        else:
            input_shape = mdp_info.observation_space.shape

        # PASSARE LA LOSS A REGRESSOR CHE DEVE ASPETTARSI UN APPROSSIMATORE CHE
        # ACCETTA LOSS COME PARAMETRO

        self._V = Regressor(value_function_approximator,
                            input_shape=input_shape, output_shape=(1,))

        self._policy_list = [deepcopy(policy) for _ in range(n_actors)]

        super().__init__(self._policy_list[0], mdp_info, policy_features)

    def fit(self, dataset):
        n_steps_per_actor = len(dataset) / self._n_actors
        A = np.empty((self._n_actors, n_steps_per_actor))
        df = self.mdp_info.gamma * self._lambda

        for i in range(self._n_actors):
            delta = np.empty(n_steps_per_actor)
            for j, step in enumerate(reversed(dataset)):
                s, a, r, ss, absorbing, _ = step

                s_phi = self.phi(s) if self.phi is not None else s
                s_psi = self._psi(s) if self._psi is not None else s
                ss_psi = self._psi(ss) if self._psi is not None else ss

                v = np.asscalar(self._V(s_psi))
                v_next = np.asscalar(self._V(ss_psi)) if not absorbing else 0

                delta[-j] = r + self.mdp_info.gamma * v_next - v

                A[i, -j] = (df ** np.arange(j)).dot(delta[-j:])


class TRPO(Agent):
    """
    Trust Region Policy Optimization algorithm.
    "Trust Region Policy Optimization". Schulman J. et al.. 2015.

    """
