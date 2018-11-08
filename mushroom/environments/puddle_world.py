import numpy as np
from scipy.stats import norm
from gym.envs.classic_control import rendering

from mushroom.environments import Environment, MDPInfo
from mushroom.utils.spaces import Discrete, Box


class PuddleWorld(Environment):
    """
    Puddle world as presented in:
    "Off-Policy Actor-Critic". Degris T. et al.. 2012.

    """
    def __init__(self, start=None, goal=None, goal_threshold=.1, noise=.025,
                 thrust=.05, puddle_center=None, puddle_width=None,
                 gamma=.99, horizon=5000):
        """
        Constructor.

        Args:
            start (np.array, None): starting position of the agent;
            goal (np.array, None): goal position;
            goal_threshold (float, .1): distance threshold of the agent from the
                goal to consider it reached;
            noise (float, .025): noise in actions;
            thrust (float, .05): distance walked during each action;
            puddle_center (np.array, None): center of the puddle;
            puddle_width (np.array, None): width of the puddle;

        """
        # MDP parameters
        self._start = np.array([.2, .4]) if start is None else start
        self._goal = np.array([1., 1.]) if goal is None else goal
        self._goal_threshold = goal_threshold
        self._noise = noise
        self._thrust = thrust
        puddle_center = [[.3, .6], [.4, .5], [.8, .9]] if puddle_center is None else puddle_center
        self._puddle_center = [np.array(center) for center in puddle_center]
        puddle_width = [[.1, .03], [.03, .1], [.03, .1]] if puddle_width is None else puddle_width
        self._puddle_width = [np.array(width) for width in puddle_width]

        self._actions = [np.zeros(2) for _ in range(5)]
        for i in range(4):
            self._actions[i][i // 2] = thrust * (i % 2 * 2 - 1)

        self._viewer = None

        # MDP properties
        action_space = Discrete(5)
        observation_space = Box(0., 1., shape=(2,))
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

    def reset(self, state=None):
        if state is None:
            self._state = self._start.copy()
        else:
            self._state = state

        return self._state

    def step(self, action):
        idx = action[0]
        self._state += self._actions[idx] + np.random.uniform(
            low=-self._noise, high=self._noise, size=(2,))
        self._state = np.clip(self._state, 0., 1.)

        reward = self._get_reward(self._state)

        absorbing = np.linalg.norm((self._state - self._goal),
                                   ord=1) < self._goal_threshold

        return self._state, reward, absorbing, {}

    def render(self, mode='human', close=False):
        if close:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            return

        screen_width = 600
        screen_height = 400

        if self._viewer is None:
            self._viewer = rendering.Viewer(screen_width, screen_height)

            import pyglet
            img_width = 100
            img_height = 100
            fformat = 'RGB'
            pixels = np.zeros((img_width, img_height, len(fformat)))
            for i in range(img_width):
                for j in range(img_height):
                    x = float(i) / img_width
                    y = float(j) / img_height
                    pixels[j, i] = self._get_reward(np.array([x, y]))

            pixels -= pixels.min()
            pixels *= 255. / pixels.max()
            pixels = np.floor(pixels)

            img = pyglet.image.create(img_width, img_height)
            img.format = fformat
            data = [chr(int(pixel)) for pixel in pixels.flatten()]

            img.set_data(fformat, img_width * len(fformat), ''.join(data))
            bg_image = Image(img, screen_width, screen_height)
            bg_image.set_color(1., 1., 1.)

            self._viewer.add_geom(bg_image)

            thickness = 5
            agent_polygon = rendering.FilledPolygon([(-thickness, -thickness),
                                                     (-thickness, thickness),
                                                     (thickness, thickness),
                                                     (thickness, -thickness)])
            agent_polygon.set_color(0., 1., 0.)
            self.agenttrans = rendering.Transform()
            agent_polygon.add_attr(self.agenttrans)
            self._viewer.add_geom(agent_polygon)

        self.agenttrans.set_translation(self._state[0] * screen_width,
                                        self._state[1] * screen_height)

        return self._viewer.render(return_rgb_array=mode == 'rgb_array')

    def stop(self):
        if self._viewer is not None:
            self._viewer.close()

    def _get_reward(self, state):
        reward = -1.
        for cen, wid in zip(self._puddle_center, self._puddle_width):
            reward -= 2. * norm.pdf(state[0], cen[0], wid[0]) * norm.pdf(
                state[1], cen[1], wid[1])

        return reward


class Image(rendering.Geom):
    def __init__(self, img, width, height):
        rendering.Geom.__init__(self)
        self.width = width
        self.height = height
        self.img = img
        self.flip = False

    def render1(self):
        self.img.blit(0, 0, width=self.width, height=self.height)
