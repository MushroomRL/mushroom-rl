import gym
import numpy as np

from PyPi.utils import spaces

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY = [(-14, 17), (-17, 0), (-17, -10), (17, -10), (17, 0), (14, 17)]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400


class LunarLander(gym.Env):
    def __init__(self, continuous=False):
        self.__name__ = 'LunarLander-v2'

        # MPD creation
        self.env = gym.make(self.__name__).env

        # MDP spaces
        high = np.array([np.inf] * 8)  # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-high, high)
        if continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,))
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        # MDP parameters
        self.horizon = 100
        self.gamma = 0.95

        # MDP properties
        self.continuous = continuous

        # MDP initialization
        self.env.seed()
        self.reset()

    def reset(self, state=None):
        if state is None:
            self.env.reset()
        else:
            self.env.state = state

        return self.get_state()

    def step(self, action):
        _, reward, absorbing, info = self.env.step(int(action[0, 0]))

        return self.get_state(), reward, absorbing, info

    def get_state(self):
        pos = self.env.lander.position
        vel = self.env.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.env.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_W / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.env.lander.angle,
            20.0 * self.env.lander.angularVelocity / FPS,
            1.0 if self.env.legs[0].ground_contact else 0.0,
            1.0 if self.env.legs[1].ground_contact else 0.0
            ]

        return np.array([state])

    def render(self, mode='human', close=False):
        self.env.render(mode=mode, close=close)

    def __str__(self):
        return self.__name__
