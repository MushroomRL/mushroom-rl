import numpy as np

from mushroom_rl.core import Core, Agent
from mushroom_rl.environments.mujoco import MultiMuJoCo
from mushroom_rl.policy import Policy
from mushroom_rl.utils.mujoco import ObservationType


def make_multi_hopper():
    xml_file = 'mushroom_rl/environments/mujoco_envs/data/hopper/model.xml'

    observation_spec = [("z_pos", "rootz", ObservationType.JOINT_POS),
                        ("y_pos", "rooty", ObservationType.JOINT_POS),
                        ("thigh_pos", "thigh_joint", ObservationType.JOINT_POS),
                        ("z_vel", "rootz", ObservationType.JOINT_VEL),
                        ("thigh_vel", "thigh_joint", ObservationType.JOINT_VEL)]
    additional_data_spec = [("x_pos", "rootx", ObservationType.JOINT_POS)]

    class MultiHopper(MultiMuJoCo):
        def reward(self, obs, action, next_obs, absorbing):
            return float(next_obs[0])

        def is_absorbing(self, obs):
            return False

    return MultiHopper([xml_file, xml_file], ["thigh_joint", "leg_joint", "foot_joint"], observation_spec,
                       0.99, 50, additional_data_spec=additional_data_spec, random_env_reset=False)


def test_multi_mujoco_reset_returns_episode_info():
    np.random.seed(1)
    mdp = make_multi_hopper()

    obs, episode_info = mdp.reset()

    assert obs.shape == mdp.info.observation_space.shape
    assert episode_info == {}


def test_multi_mujoco_runs_with_core():
    np.random.seed(2)
    mdp = make_multi_hopper()

    class ZeroPolicy(Policy):
        def draw_action(self, state):
            return np.zeros(3)

    class DummyAgent(Agent):
        def fit(self, dataset):
            pass

    dataset = Core(DummyAgent(mdp.info, ZeroPolicy()), mdp).evaluate(n_episodes=2, quiet=True)

    assert len(dataset) == 100
    assert dataset.n_episodes == 2


def test_multi_mujoco_switches_model_and_helper():
    np.random.seed(3)
    mdp = make_multi_hopper()

    mdp.reset()
    first_model, first_helper = mdp._model, mdp.obs_helper
    mdp.reset()
    second_model, second_helper = mdp._model, mdp.obs_helper

    assert first_model is not second_model
    assert first_helper is not second_helper
    assert second_helper is mdp.obs_helpers[mdp._current_model_idx]
