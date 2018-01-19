import numpy as np

from mushroom.environments import Environment, MDPInfo
from mushroom.utils import spaces


class LQR(Environment):
    """
    This class implements a Linear-Quadratic Regulator.
    This task aims to minimize the undesired deviations from nominal values of
    some controller settings in control problems.
    The system equations in this task are:

    .. math::
        x_{t+1} = Ax_t + Bu_t

    where x is the state and u is the control signal.

    The reward function is given by:

    .. math::
        r_t = -\\left( x_t^TQx_t + u_t^TRu_t \\right)

    "Policy gradient approaches for multi-objective sequential decision making".
    Parisi S., Pirotta M., Smacchia N., Bascetta L., Restelli M.. 2014

    """
    def __init__(self, A, B, Q, R, random_init=False, gamma=0.9, horizon=50):
        """
        Constructor.

            Args:
                A (np.ndarray): the state dynamics matrix;
                B (np.ndarray): the action dynamics matrix;
                Q (np.ndarray): reward weight matrix for state;
                R (np.ndarray): reward weight matrix for action;
                random_init (bool, False): start from a random state;
                gamma (float, 0.9): discount factor;
                horizon (int, 50): horizon of the mdp.

        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.random_init = random_init

        # MDP properties
        high_x = np.inf * np.ones(A.shape[0])
        low_x = -high_x

        high_u = np.inf * np.ones(B.shape[0])
        low_u = -high_u

        observation_space = spaces.Box(low=low_x, high=high_x)
        action_space = spaces.Box(low=low_u, high=high_u)
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super(LQR, self).__init__(mdp_info)

    @staticmethod
    def generate(dimensions, eps=0.1, index=0, random_init=False,
                 gamma=0.9, horizon=50):
        """
        Factory method that generates an lqr with identity dynamics and
        symmetric reward matrices.

        Args:
            dimensions (int): number of state/action dimensions;
            eps (double, 0.1): reward matrix weights specifier;
            index (int, 0): selector for the principal state;
            random_init (bool, False): start from a random state;
            gamma (float, 0.9): discount factor;
            horizon (int, 50): horizon of the mdp.

        """
        assert dimensions >= 1

        A = np.eye(dimensions)
        B = np.eye(dimensions)
        Q = eps * np.eye(dimensions)
        R = (1. - eps) * np.eye(dimensions)

        Q[index, index] = 1. - eps
        R[index, index] = eps

        return LQR(A, B, Q, R, random_init, gamma, horizon)

    def reset(self, state=None):
        if state is None:
            if self.random_init:
                self._state = np.random.uniform(-3, 3, size=self.A.shape[0])
            else:
                self._state = 10. * np.ones(self.A.shape[0])
        else:
            self._state = state

        return self._state

    def step(self, action):
        x = self._state
        u = action

        reward = -(x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))
        self._state = self.A.dot(x) + self.B.dot(u)

        return self._state, reward, False, {}
