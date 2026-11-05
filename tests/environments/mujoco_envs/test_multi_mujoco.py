import numpy as np
import pytest
import torch

from mushroom_rl.core import Core, Agent
from mushroom_rl.environments import MuJoCo, MultiMuJoCo
from mushroom_rl.policy import Policy

mujoco = pytest.importorskip('mujoco')

from mushroom_rl.utils.mujoco import (ObservationHelper, ObservationType, MuJoCoObservationHelper,  # noqa: E402
                                      WarpObservationHelper)


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
    return ('mushroom_rl/environments/mujoco_envs/data/hopper/model.xml', ["thigh_joint", "leg_joint", "foot_joint"],
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


def make_worlds(mjw, wp):
    spec = mujoco.MjSpec.from_file('mushroom_rl/environments/mujoco_envs/data/ant/model.xml')
    spec.body('torso').add_site(name='probe', pos=[.1, .2, .3])
    model = spec.compile()
    observation_spec = [('root_pose', 'root', ObservationType.JOINT_POS), ('hip', 'hip_1', ObservationType.JOINT_POS),
                        ('root_vel', 'root', ObservationType.JOINT_VEL),
                        ('leg_pos', 'front_left_leg', ObservationType.BODY_POS),
                        ('leg_rot', 'front_left_leg', ObservationType.BODY_ROT),
                        ('leg_vel_world', 'front_left_leg', ObservationType.BODY_VEL_WORLD),
                        ('leg_vel', 'front_left_leg', ObservationType.BODY_VEL),
                        ('site_pos', 'probe', ObservationType.SITE_POS),
                        ('site_rot', 'probe', ObservationType.SITE_ROT)]

    rng = np.random.RandomState(0)
    datas = list()
    for _ in range(2):
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0 + .1 * rng.randn(model.nq)
        data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
        data.qvel[:] = rng.randn(model.nv)
        mujoco.mj_forward(model, data)
        datas.append(data)

    data_wp = mjw.put_data(model, datas[0], nworld=2)
    wp.to_torch(data_wp.qpos)[1] = torch.as_tensor(datas[1].qpos, dtype=torch.float32)
    wp.to_torch(data_wp.qvel)[1] = torch.as_tensor(datas[1].qvel, dtype=torch.float32)
    mjw.forward(mjw.put_model(model), data_wp)

    return model, observation_spec, datas, data_wp


def test_warp_observation_helper():
    mjw = pytest.importorskip('mujoco_warp')
    wp = pytest.importorskip('warp')
    wp.set_device('cpu')
    model, observation_spec, datas, data_wp = make_worlds(mjw, wp)
    helper = WarpObservationHelper(observation_spec, model, data_wp, None)
    references = [MuJoCoObservationHelper(observation_spec, model, data, None) for data in datas]
    obs = helper.build_obs()

    assert obs.shape == (2, 45)
    for world, reference in enumerate(references):
        assert np.allclose(obs[world].numpy(), reference.build_obs(), atol=1e-6)
    for _, name, o_type in observation_spec:
        expected = np.asarray(references[1].get_state(name, o_type)).reshape(-1)
        for selection in ([1], torch.tensor([1])):
            assert np.allclose(helper.get_state(name, o_type, selection).numpy().reshape(-1), expected, atol=1e-6)
        assert helper.get_state(name, o_type).shape[0] == 2

    helper.set_state('hip_1', ObservationType.JOINT_POS, torch.tensor([[.3], [.4]]))
    helper.set_state('hip_1', ObservationType.JOINT_VEL, np.array([[-1.]]), torch.tensor([0]))

    assert torch.allclose(helper.get_state('hip_1', ObservationType.JOINT_POS), torch.tensor([[.3], [.4]]))
    assert helper.get_state('hip_1', ObservationType.JOINT_VEL)[0, 0] == -1.

    modified = obs.clone()
    modified[:, helper.obs_idx_map['hip']] = torch.tensor([[.7], [.8]])
    modified[:, helper.obs_idx_map['leg_pos']] = 100.
    helper.modify_data(modified)

    assert torch.allclose(helper.get_state('hip_1', ObservationType.JOINT_POS), torch.tensor([[.7], [.8]]))
    assert torch.equal(helper.get_state('front_left_leg', ObservationType.BODY_POS),
                       obs[:, helper.obs_idx_map['leg_pos']])

    with pytest.raises(ValueError):
        helper.set_state('front_left_leg', ObservationType.BODY_POS, torch.zeros(2, 3))
    with pytest.raises(ValueError):
        helper.get_state('front_left_leg', 'unknown')


def test_observation_helper():
    spec = mujoco.MjSpec.from_file('mushroom_rl/environments/mujoco_envs/data/ant/model.xml')
    body = spec.worldbody.add_body(name='ball_body', pos=[0, 0, 2])
    body.add_joint(name='ball', type=mujoco.mjtJoint.mjJNT_BALL)
    body.add_geom(size=[.1, 0, 0])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    helper = MuJoCoObservationHelper([('ball_pos', 'ball', ObservationType.JOINT_POS),
                                      ('ball_vel', 'ball', ObservationType.JOINT_VEL),
                                      ('root_pose', 'root', ObservationType.JOINT_POS)], model, data, None)
    helper.add_obs('extra', 2, np.zeros(2), np.ones(2))
    low, high = helper.get_obs_limits()

    assert [len(helper.obs_idx_map[key]) for key in ('ball_pos', 'ball_vel', 'root_pose', 'extra')] == [4, 3, 7, 2]
    assert np.array_equal(low[-2:], [0., 0.]) and np.array_equal(high[-2:], [1., 1.])

    with pytest.raises(AttributeError):
        MuJoCoObservationHelper([], model, data, None)
    for call in (lambda: helper._address('root', 'unknown'), lambda: helper._obs_size('root', 'unknown')):
        with pytest.raises(ValueError):
            call()
    for call in (lambda: ObservationHelper.build_obs(None), lambda: ObservationHelper.modify_data(None, None),
                 lambda: ObservationHelper.get_state(None, None, None),
                 lambda: ObservationHelper.set_state(None, None, None, None)):
        with pytest.raises(NotImplementedError):
            call()
