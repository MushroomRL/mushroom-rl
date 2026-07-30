import torch
import numpy as np

from mushroom_rl.policy.policy import Policy, StatefulPolicy, HasWeights


class OrnsteinUhlenbeckPolicy(HasWeights, StatefulPolicy):
    """
    Exploration policy adding temporally correlated noise drawn from an Ornstein-Uhlenbeck process.

    This policy is commonly used in the Deep Deterministic Policy Gradient algorithm. The process is implemented
    as in https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py.

    """
    def __init__(self, mu, sigma, theta, dt, x0=None):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the state;
            sigma (torch.tensor): average magnitude of the random fluctations per square-root time;
            theta (float): rate of mean reversion;
            dt (float): time interval;
            x0 (torch.tensor, None): initial values of noise.

        """
        super().__init__(policy_state_shape=tuple(mu.output_shape))

        self._approximator = mu
        self._predict_params = dict()
        self._sigma = sigma
        self._theta = theta
        self._dt = dt
        self._x0 = x0

        self._add_save_attr(
            _approximator='mushroom',
            _predict_params='pickle',
            _sigma='torch',
            _theta='primitive',
            _dt='primitive',
            _x0='torch'
        )

    def __call__(self, state, action=None, policy_state=None):
        raise NotImplementedError

    def _draw_action(self, state, policy_state):
        with torch.no_grad():
            mu = self._approximator.predict(state, **self._predict_params)
            sqrt_dt = np.sqrt(self._dt)

            x = policy_state - self._theta * policy_state * self._dt +\
                self._sigma * sqrt_dt * torch.randn_like(policy_state)

            return mu + x, x

    def _draw_action_greedy(self, state, policy_state):
        with torch.no_grad():
            mu = self._approximator.predict(state, **self._predict_params)

            return mu, policy_state

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size

    def reset(self):
        self._policy_state = self._x0.clone() if self._x0 is not None else torch.zeros(self._approximator.output_shape)

        return self._policy_state

    def reset_vectorized(self, start_mask):
        if self._policy_state is None:
            self._policy_state = torch.zeros((len(start_mask),) + tuple(self._approximator.output_shape))
        self._policy_state[start_mask] = self._x0 if self._x0 is not None else 0.

        return self._policy_state


class ClippedGaussianPolicy(HasWeights, Policy):
    """
    Gaussian policy whose sampled action is clipped to a given action range.

    This policy was introduced used in:
    "Addressing Function Approximation Error in Actor-Critic Methods".
    Fujimoto S. et al. 2018.

    This is a non-differentiable policy for continuous action spaces.
    The policy samples an action in every state following a gaussian distribution, where the mean is computed in the
    state and the covariance matrix is fixed. The action is then clipped using the given action range.
    This policy is not a truncated Gaussian, as it simply clips the action if the value is bigger than the boundaries.
    Thus, the non-differentiability.

    """
    def __init__(self, mu, sigma, low, high):
        """
        Constructor.

        Args:
            mu (Regressor): the regressor representing the mean w.r.t. the state;
            sigma (torch.tensor): a square positive definite matrix representing the covariance matrix. The size of this
                matrix must be n x n, where n is the action dimensionality;
            low (torch.tensor): a vector containing the minimum action for each component;
            high (torch.tensor): a vector containing the maximum action for each component.

        """
        self._approximator = mu
        self._predict_params = dict()
        self._chol_sigma = torch.linalg.cholesky(sigma)
        self._low = torch.as_tensor(low)
        self._high = torch.as_tensor(high)

        self._add_save_attr(
            _approximator='mushroom',
            _predict_params='pickle',
            _chol_sigma='torch',
            _low='torch',
            _high='torch'
        )

    def __call__(self, state, action=None):
        raise NotImplementedError

    def draw_action(self, state):
        with torch.no_grad():
            mu = self._approximator.predict(state, **self._predict_params).reshape(-1)

            distribution = torch.distributions.MultivariateNormal(loc=mu, scale_tril=self._chol_sigma,
                                                                  validate_args=False)

            action_raw = distribution.sample()

            return torch.clip(action_raw, self._low, self._high)

    def draw_action_greedy(self, state):
        with torch.no_grad():
            mu = self._approximator.predict(state, **self._predict_params).reshape(-1)

            return torch.clip(mu, self._low, self._high)

    def set_weights(self, weights):
        self._approximator.set_weights(weights)

    def get_weights(self):
        return self._approximator.get_weights()

    @property
    def weights_size(self):
        return self._approximator.weights_size
