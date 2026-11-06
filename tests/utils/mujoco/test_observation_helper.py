import numpy as np
import pytest
import torch

mujoco = pytest.importorskip('mujoco')

from mushroom_rl.utils.mujoco import (ObservationHelper, ObservationType, MuJoCoObservationHelper,  # noqa: E402
                                      WarpObservationHelper)


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
    import warp as wp
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
