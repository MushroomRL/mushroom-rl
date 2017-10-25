import numpy as np

from mushroom.environments import Environment
from mushroom.utils import spaces


class ShipSteering(Environment):

    def __init__(self, small=True):
        self.__name__ = 'ShipSteering'

        # MDP spaces
        self.fieldSize = 150 if small else 1000
        low = np.array([0, 0, -np.pi, -np.pi / 12.0])
        high = np.array([self.fieldSize, self.fieldSize, np.pi, np.pi/12.0])
        self.omegaMax = np.array([np.pi/12.0])
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Box(low=-self.omegaMax, high=self.omegaMax)

        # MDP parameters
        self.horizon = 5000
        self.gamma = 0.99

        # MDP properties
        self._v = 3.0
        self._T = 5.0
        self._dt = .2

        self.gateS = np.empty(2)
        self.gateE = np.empty(2)
        self.gateS[0] = 100 if small else 900
        self.gateS[1] = 120 if small else 920
        self.gateE[0] = 120 if small else 920
        self.gateE[1] = 100 if small else 900


        super(ShipSteering, self).__init__()

    def reset(self, state=None):
        if state is None:
            self._state = np.array([0, 0, 0, 0])
        else:
            self._state = state

        return self._state

    def step(self, action):
        sa = np.append(self._state, action)
        r = max(-self.omegaMax, min(self.omegaMax, action[0]))
        new_state = np.empty(4)
        new_state[0] = self._state[0]+self._v*np.sin(self._state[2])*self._dt
        new_state[1] = self._state[1]+self._v*np.cos(self._state[2])*self._dt
        new_state[2] = self._state[2]+self._state[3]*self._dt
        new_state[3] = self._state[3]+(r-self._state[3])*self._dt/self._T

        if new_state[0] > self.fieldSize or new_state[1] > self.fieldSize or new_state[0] < 0 or new_state[1] < 0:
            reward = -100
            absorbing = True

        elif self._throughGate(self._state[:2], new_state[:2]):
            reward = 0
            absorbing = True

        else:
            reward = -1
            absorbing = False

        self._state = new_state

        return self._state, reward, absorbing, {}

    def _throughGate(self, start, end):
        r=self.gateE-self.gateS
        s=end-start
        den= self._cross2D(vecr=r, vecs=s)


        if den==0:
            return False

        t = self._cross2D((start-self.gateS),s)/den
        u = self._cross2D((start-self.gateS),r)/den

        return u>=0 and u<=1 and t>=0 and t<=1

    def _cross2D(self,vecr,vecs):
        return vecr[0]*vecs[1]-vecr[1]*vecs[0]
