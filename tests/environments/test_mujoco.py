import numpy as np
import pytest

from mushroom_rl.core import Core, Agent
from mushroom_rl.environments import MuJoCo, MultiMuJoCo
from mushroom_rl.environments.assets import asset_path
from mushroom_rl.policy import Policy

mujoco = pytest.importorskip('mujoco')

from mushroom_rl.utils.mujoco import ObservationType  # noqa: E402


def make_multi_hopper():
    xml_file = asset_path('hopper', 'model.xml')

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


def test_multi_mujoco():
    np.random.seed(1)
    mdp = make_multi_hopper()

    obs, episode_info = mdp.reset()

    assert obs.shape == mdp.info.observation_space.shape
    assert episode_info == {}

    np.random.seed(2)
    mdp = make_multi_hopper()

    class ZeroPolicy(Policy):
        def draw_action(self, state):
            return np.zeros(3)

    class DummyAgent(Agent):
        def fit(self, dataset):
            pass

    dataset = Core(DummyAgent(mdp.info, ZeroPolicy('numpy')), mdp).evaluate(n_episodes=2, quiet=True)

    assert len(dataset) == 100
    assert dataset.n_episodes == 2

    np.random.seed(3)
    mdp = make_multi_hopper()

    mdp.reset()
    first_model, first_helper = mdp._model, mdp.obs_helper
    mdp.reset()
    second_model, second_helper = mdp._model, mdp.obs_helper

    assert first_model is not second_model
    assert first_helper is not second_helper
    assert second_helper is mdp.obs_helpers[mdp._current_model_idx]


def make_spec():
    return (asset_path('hopper', 'model.xml'), ["thigh_joint", "leg_joint", "foot_joint"],
            [("z_pos", "rootz", ObservationType.JOINT_POS), ("thigh_pos", "thigh_joint", ObservationType.JOINT_POS),
             ("z_vel", "rootz", ObservationType.JOINT_VEL)])


def make_class(base):
    class Counting(base):
        def __init__(self, *args, **kwargs):
            self.calls = 0
            super().__init__(*args, **kwargs)

        def _compute_action(self, obs, action):
            self.calls += 1
            return action

        def reward(self, obs, action, next_obs, absorbing):
            return 0.

        def is_absorbing(self, obs):
            return False

    return Counting


def test_mujoco():
    xml, actuation, observation = make_spec()
    mdp = MuJoCo(xml, actuation, observation, .99, 50)

    assert mdp.get_all_observation_keys() == ['z_pos', 'thigh_pos', 'z_vel']
    assert np.array_equal(mdp.reset(np.array([1.5, -.2, .3]))[0], [1.5, -.2, .3])
    assert MuJoCo(xml, [], observation, .99, 50).info.action_space.shape == (3,)

    with pytest.raises(NotImplementedError):
        mdp.reward(None, None, None, None)
    with pytest.raises(NotImplementedError):
        mdp.is_absorbing(None)

    mdp.seed(5)
    first = np.random.rand()
    mdp.seed(5)

    assert np.random.rand() == first

    spec = mujoco.MjSpec.from_file(xml)
    spec.actuators[0].ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    action_space = MuJoCo(spec, actuation, observation, .99, 50).info.action_space

    assert np.array_equal(action_space.low, [-np.inf, -1., -1.])
    assert np.array_equal(action_space.high, [np.inf, 1., 1.])

    with pytest.raises(ValueError):
        MuJoCo(3, actuation, observation, .99, 50)

    for warning, message in [('Pre-allocated constraint buffer is full',
                              'Pre-allocated constraint buffer is full: Increase njmax in mujoco XML'),
                             ('Pre-allocated contact buffer is full',
                              'Pre-allocated contact buffer is full: Increase njconmax in mujoco XML'),
                             ('Unknown warning type', 'Unknown warning type: Check for NaN in simulation.'),
                             ('other', 'Got MuJoCo Warning: other')]:
        with pytest.raises(RuntimeError) as error:
            MuJoCo.user_warning_raise_exception(warning)

        assert str(error.value) == message

    mdp = make_class(MuJoCo)(xml, actuation, observation, .99, 50, n_intermediate_steps=2)
    mdp.reset()
    mdp.step(np.zeros(3))

    assert mdp.calls == 2


def test_multi_options():
    xml, actuation, observation = make_spec()
    np.random.seed(1)
    mdp = make_class(MultiMuJoCo)([xml] * 3, actuation, observation, .99, 50, timestep=.004, n_intermediate_steps=2)
    models = list()
    for _ in range(6):
        mdp.reset()
        models.append(mdp._current_model_idx)
    mdp.step(np.zeros(3))

    assert models == [1, 0, 0, 1, 1, 0]
    assert mdp.calls == 2
    assert mdp.more_than_one_env
    assert mdp._model.opt.timestep == .004 and mdp.dt == .008
    assert [MultiMuJoCo._get_env_id_map(i, n).tolist() for i, n in [(0, 1), (1, 2), (2, 4), (3, 4), (5, 8)]] == \
        [[0.], [1.], [1., 0.], [1., 1.], [1., 0., 1.]]

    action_spec = mujoco.MjSpec.from_file(xml)
    action_spec.actuators[0].ctrlrange = [-2., 2.]
    limits_spec = mujoco.MjSpec.from_file(xml)
    limits_spec.joint('thigh_joint').range = [-100., 0.]

    for spec in (action_spec, limits_spec):
        with pytest.raises(ValueError):
            make_class(MultiMuJoCo)([xml, spec], actuation, observation, .99, 50)
