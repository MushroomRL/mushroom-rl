import numpy as np

from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.rl_utils import spaces


class FiniteMDP(Environment):
    """
    Finite Markov Decision Process.

    """
    def __init__(self, p, rew, mu=None, gamma=.9, horizon=np.inf, dt=1e-1):
        """
        Constructor.

        Args:
            p (np.ndarray): transition probability matrix;
            rew (np.ndarray): reward matrix;
            mu (np.ndarray, None): initial state probability distribution;
            gamma (float, .9): discount factor;
            horizon (int, np.inf): the horizon;
            dt (float, 1e-1): the control timestep of the environment.

        """
        assert p.shape == rew.shape
        assert mu is None or p.shape[0] == mu.size

        # MDP parameters
        self.p = p
        self.r = rew
        self.mu = mu

        # MDP properties
        observation_space = spaces.Discrete(p.shape[0])
        action_space = spaces.Discrete(p.shape[1])
        horizon = horizon
        gamma = gamma
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, dt)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            if self.mu is not None:
                self._state = np.array(
                    [np.random.choice(self.mu.size, p=self.mu)])
            else:
                self._state = np.array([np.random.choice(self.p.shape[0])])
        else:
            self._state = state

        return self._state, {}

    def step(self, action):
        p = self.p[self._state[0], action[0], :]
        next_state = np.array([np.random.choice(p.size, p=p)])
        absorbing = not np.any(self.p[next_state[0]])
        reward = self.r[self._state[0], action[0], next_state[0]]

        self._state = next_state

        return self._state, reward, absorbing, {}
