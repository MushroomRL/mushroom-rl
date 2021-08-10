import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from mj_envs.hand_manipulation_suite import *
    from mjrl.envs import *

import gym
from mushroom_rl.core import Environment, MDPInfo
from mushroom_rl.utils.spaces import *

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def _get_embedding(obs_size=3, embedding_name='baseline', train=False):
    if embedding_name == 'baseline':
        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        return nn.Sequential(
            init_(nn.Conv2d(obs_size, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)),
            nn.ELU(),
        )
    elif embedding_name == 'resnet34':
        model = models.resnet34(pretrained=not train)
        layers = list(model.children())[:-1]
        return nn.Sequential(*layers)
    else:
        raise NotImplementedError


class StateEmbedding(gym.ObservationWrapper):
    def __init__(self, env, embedding_name='baseline', train=False):
        gym.ObservationWrapper.__init__(self, env)
        original_obs_space = env.observation_space

        embedding = _get_embedding(embedding_name=embedding_name)

        if not train:
            for p in embedding.parameters():
                p.requires_grad = False

        dummy_obs = torch.zeros(1, *original_obs_space.shape)
        embedding_space_shape = np.prod(embedding(dummy_obs).shape)

        self.embedding = embedding
        self.train = train
        self.observation_space = Box(
                    low=-np.inf, high=np.inf, shape=(embedding_space_shape,))

    def observation(self, observation):
        # From https://pytorch.org/vision/stable/models.html
        # All pre-trained models expect input images normalized in the same way,
        # i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
        # where H and W are expected to be at least 224.
        # The images have to be loaded in to a range of [0, 1] and then
        # normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
        observation = torch.from_numpy(np.ascontiguousarray(observation[None,:,:,:], dtype=np.float32))
        observation /= 255.
        # transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        # observation = transforms
        if self.train:
            return self.embedding(observation).view(1, -1).squeeze()
            raise NotImplementedError
        else:
            with torch.no_grad():
                return self.embedding(observation).view(1, -1).numpy().squeeze()


class MuJoCoPixelObs(gym.ObservationWrapper):
    def __init__(self, env, width, height, camera_name, camera_id=0, depth=False):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0., high=255., shape=(3, width, height))
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.depth = depth
        self.camera_id = camera_id

    def observation(self, observation):
        img = self.sim.render(width=self.width, height=self.height, depth=self.depth,
                              camera_name=self.camera_name, device_id=self.camera_id)
        img = img[::-1,:,:]
        return img.transpose((2, 0, 1))


class MJEnv(Environment):
    """

    """
    def __init__(self, task_name, horizon=1000, gamma=0.99,
        use_pixels=False, camera_id=0, pixels_width=64, pixels_height=64,
        embedding='baseline'):
        """
        Constructor.

        Args:
             task_name (str): name of the task;
             horizon (int): the horizon;
             gamma (float): the discount factor;
             use_pixels (bool, False): if True, pixel observations are used
                rather than the state vector;
             camera_id (int, 0): position of camera to render the environment;
             pixels_width (int, 64): width of the pixel observation;
             pixels_height (int, 64): height of the pixel observation;

        """

        self.env = gym.make(task_name)
        if use_pixels:
            self.env = MuJoCoPixelObs(self.env, width=pixels_width,
                                                height=pixels_height,
                                                camera_name='vil_camera',
                                                camera_id=camera_id)
            self.env = StateEmbedding(self.env, embedding)

        # MDP properties
        action_space = self.env.action_space
        observation_space = self.env.observation_space
        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon)

        super().__init__(mdp_info)

        self._state = None

    def reset(self, state=None):
        if state is None:
            return self.env.reset()
        else:
            raise NotImplementedError

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._state = obs
        return obs, reward, done, info

    def render(self):
        pass

    def stop(self):
        pass
