import mujoco
import mujoco_warp as mj_warp
import torch

import warp as wp

from mushroom_rl.core import VectorizedEnvironment, MDPInfo
from mushroom_rl.core.spaces import Box
from mushroom_rl.utils.mujoco import ObservationHelper, ObservationType, MujocoViewer
from mushroom_rl.utils.torch_utils import TorchUtils
from mushroom_rl.environments.mujoco import MuJoCo


class MuJoCoWarp(VectorizedEnvironment):
    """
    Class to create N parallel environments using MuJoCo Warp for GPU-accelerated batch simulation.

    """

    def __init__(
        self,
        xml_file,
        actuation_spec,
        observation_spec,
        gamma,
        horizon,
        num_envs,
        timestep=None,
        n_substeps=1,
        n_intermediate_steps=1,
        additional_data_spec=None,
        collision_groups=None,
        max_joint_vel=None,
        nconmax=None,
        njmax=None,
        use_graph_capture=False,
        warmup_steps=3,
        **viewer_params,
    ):
        self._mj_warp = mj_warp
        self._wp = wp

        self._model = MuJoCo.load_model(xml_file)
        if timestep is not None:
            self._model.opt.timestep = timestep
            self._timestep = timestep
        else:
            self._timestep = self._model.opt.timestep

        self._num_envs = num_envs
        self._n_intermediate_steps = n_intermediate_steps
        self._n_substeps = n_substeps
        self._viewer_params = viewer_params
        self._viewer = None
        self._obs = None

        self._use_graph_capture = use_graph_capture
        self._warmup_steps = warmup_steps
        self._sim_step_graph = None

        self._model_wp = mj_warp.put_model(self._model)
        self._data_wp = mj_warp.make_data(
            self._model, nworld=num_envs, nconmax=nconmax, njmax=njmax
        )

        self._reset_mask_t = torch.zeros(
            self._num_envs, dtype=torch.bool, device=TorchUtils.get_device()
        )
        self._reset_mask_wp = wp.from_torch(self._reset_mask_t)

        self._reset_graph = None
        self._reset_build_count = 0  # diagnostic
        self._reset_launch_count = 0  # diagnostic

        _tmp_data = mujoco.MjData(self._model)
        self._action_indices = MuJoCo.get_action_indices(
            self._model, _tmp_data, actuation_spec
        )
        action_space = MuJoCo.get_action_space(self._action_indices, self._model)

        self.obs_helper = ObservationHelper(
            observation_spec,
            self._model,
            _tmp_data,
            max_joint_velocity=max_joint_vel,
            is_warp=True,
        )
        observation_space = Box(*self.obs_helper.get_obs_limits())

        self.additional_data = {}
        if additional_data_spec is not None:
            for key, name, ot in additional_data_spec:
                self.additional_data[key] = (name, ot)

        self.collision_groups = {}
        if collision_groups is not None:
            for name, geom_names in collision_groups:
                col_group = []
                for geom_name in geom_names:
                    mj_id = mujoco.mj_name2id(
                        self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
                    )
                    assert (
                        mj_id != -1
                    ), f'geom "{geom_name}" not found! Can\'t be used for collision-checking.'
                    col_group.append(mj_id)
                self.collision_groups[name] = set(col_group)

        mdp_info = MDPInfo(
            observation_space, action_space, gamma, horizon, self.dt, backend="torch"
        )
        mdp_info = self._modify_mdp_info(mdp_info)

        mujoco.set_mju_user_warning(MuJoCo.user_warning_raise_exception)

        self._recompute_action_per_step = (
            type(self)._compute_action != MuJoCoWarp._compute_action
        )

        # Cache action_indices as torch tensor on the target device
        self._action_indices_t = torch.as_tensor(
            self._action_indices, device=TorchUtils.get_device(), dtype=torch.long
        )

        super().__init__(mdp_info, num_envs)

    # ------------------------------------------------------------------
    # VectorizedEnvironment interface
    # ------------------------------------------------------------------

    def step_all(self, env_mask, action):
        action = self._preprocess_action(action)
        if isinstance(action, torch.Tensor) and action.dim() == 1:
            action = action.unsqueeze(0)
        self._step_init(self._obs, action)

        ctrl_action = None

        for i in range(self._n_intermediate_steps):
            if self._recompute_action_per_step or ctrl_action is None:
                ctrl_action = self._compute_action(self._obs, action)
                self._set_ctrl(ctrl_action, env_mask)

            self._simulation_pre_step()

            self.step_graph()

            self._simulation_post_step()

        cur_obs = self._create_observation(self.obs_helper.build_obs(self._data_wp))

        self._step_finalize()

        absorbing = self.is_absorbing(cur_obs)
        reward = self.reward(self._obs, action, cur_obs, absorbing)
        info = self._create_info_dictionary(cur_obs)

        self._obs = cur_obs.clone()

        out_device = TorchUtils.get_device()

        return (
            self._modify_observation(cur_obs).to(out_device),
            reward.to(out_device),
            (absorbing & env_mask).to(out_device),
            info,
        )

    def step_graph(self):
        if not self._use_graph_capture:
            for _ in range(self._n_substeps):
                self._mj_warp.step(self._model_wp, self._data_wp)
            return
        if self._sim_step_graph is None:
            with self._wp.ScopedCapture() as cap:
                for _ in range(self._n_substeps):
                    self._mj_warp.step(self._model_wp, self._data_wp)
            self._sim_step_graph = cap.graph

        self._wp.capture_launch(self._sim_step_graph)

    def reset_all(self, env_mask, state=None):
        env_indices = torch.where(env_mask)[0]

        self._reset_mask_t.zero_()
        self._reset_mask_t[env_indices] = True
        self._reset_data(self._reset_mask_wp)
        self.setup(env_indices, state)

        obs = self._create_observation(self.obs_helper.build_obs(self._data_wp))
        obs = self._modify_observation(obs)

        if self._obs is None:
            self._obs = obs.clone()
        else:
            self._obs[env_mask] = obs[env_mask]

        info = self._create_info_dictionary(obs)
        return obs.clone(), info

    def _reset_data(self, reset_mask_wp):
        if not self._use_graph_capture:
            self._mj_warp.reset_data(self._model_wp, self._data_wp, reset=reset_mask_wp)
            return

        if self._reset_graph is None:
            print(f"[reset] building graph")
            with self._wp.ScopedCapture() as cap:
                self._mj_warp.reset_data(
                    self._model_wp, self._data_wp, reset=reset_mask_wp
                )
            self._reset_graph = cap.graph
            print(f"[reset] graph built")

        self._wp.capture_launch(self._reset_graph)

    def render_all(self, env_mask, record=False):
        if self._viewer is None:
            self._viewer = MujocoViewer(
                self._model, self.dt, record=record, **self._viewer_params
            )

        render_data = mujoco.MjData(self._model)
        self._mj_warp.get_data_into(
            render_data, self._model, self._data_wp, world_id=self._default_env
        )

        return self._viewer.render(render_data, record)

    def stop(self):
        if self._viewer is not None:
            self._viewer.stop()
            del self._viewer
            self._viewer = None

    def seed(self, seed):

        torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    def reward(self, obs, action, next_obs, absorbing):
        raise NotImplementedError

    def is_absorbing(self, obs):
        raise NotImplementedError

    def setup(self, env_indices, obs):
        if obs is not None:
            self.obs_helper._modify_warp_data(self._data_wp, obs, env_indices)

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _modify_mdp_info(self, mdp_info):
        return mdp_info

    def _create_observation(self, obs):
        return obs

    def _modify_observation(self, obs):
        return obs

    def _create_info_dictionary(self, obs):
        return {}

    def _preprocess_action(self, action):
        return action

    def _compute_action(self, obs, action):
        return action

    def _step_init(self, obs, action):
        pass

    def _simulation_pre_step(self):
        pass

    def _simulation_post_step(self):
        pass

    def _step_finalize(self):
        pass

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _set_ctrl(self, ctrl_action, env_mask):
        """
        Write ctrl_action into the warp data for active environments.
        Inactive environments (env_mask=False) keep their previous control signal.
        """
        ctrl = wp.to_torch(self._data_wp.ctrl)
        device = ctrl.device

        ctrl_action_t = (
            ctrl_action
            if isinstance(ctrl_action, torch.Tensor)
            else torch.as_tensor(ctrl_action, device=device, dtype=ctrl.dtype)
        )
        ctrl_action_t = ctrl_action_t.to(device=device, dtype=ctrl.dtype)

        action_indices_t = self._action_indices_t.to(device)
        env_indices = torch.nonzero(env_mask.to(device), as_tuple=True)[0]

        ctrl[env_indices.unsqueeze(1), action_indices_t.unsqueeze(0)] = ctrl_action_t[
            env_indices
        ]

    def _read_data(self, name, env_indices=None):
        field_name, ot = self.additional_data[name]
        data = self._read_warp_field(field_name, ot)
        if env_indices is not None:
            if not isinstance(env_indices, torch.Tensor):
                env_indices = torch.as_tensor(
                    env_indices, device=data.device, dtype=torch.long
                )
            return data[env_indices]
        return data

    def _write_data(self, name, value, env_indices=None):
        field_name, ot = self.additional_data[name]
        device = TorchUtils.get_device()

        if env_indices is None:
            env_indices = torch.arange(self._num_envs, device=device, dtype=torch.long)

        value_t = (
            value
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, device=device)
        )

        if ot == ObservationType.JOINT_POS:
            jnt = self._model.joint(field_name)
            adr = self._model.jnt_qposadr[jnt.id]
            size = ObservationHelper._obs_size(self._model, field_name, ot)
            qpos = wp.to_torch(self._data_wp.qpos)
            col_idx = torch.arange(adr, adr + size, device=device, dtype=torch.long)
            qpos[env_indices.unsqueeze(1), col_idx.unsqueeze(0)] = value_t.to(
                qpos.dtype
            )
        elif ot == ObservationType.JOINT_VEL:
            jnt = self._model.joint(field_name)
            adr = self._model.jnt_dofadr[jnt.id]
            size = ObservationHelper._obs_size(self._model, field_name, ot)
            qvel = wp.to_torch(self._data_wp.qvel)
            col_idx = torch.arange(adr, adr + size, device=device, dtype=torch.long)
            qvel[env_indices.unsqueeze(1), col_idx.unsqueeze(0)] = value_t.to(
                qvel.dtype
            )
        else:
            raise ValueError(
                f"_write_data only supports JOINT_POS and JOINT_VEL; got {ot}."
            )

    def _read_warp_field(self, name, ot):
        """Return a torch tensor for a given named object and observation type."""
        if ot == ObservationType.BODY_POS:
            return wp.to_torch(self._data_wp.xpos)[:, self._model.body(name).id, :]
        elif ot == ObservationType.BODY_ROT:
            return wp.to_torch(self._data_wp.xquat)[:, self._model.body(name).id, :]

        elif ot == ObservationType.BODY_VEL_WORLD:
            body_id = self._model.body(name).id
            root_id = self._model.body_rootid[body_id]
            cvel = wp.to_torch(self._data_wp.cvel)[:, body_id, :]
            xpos = wp.to_torch(self._data_wp.xpos)[:, body_id, :]
            subtree_com = wp.to_torch(self._data_wp.subtree_com)[:, root_id, :]
            offset = xpos - subtree_com
            lin = cvel[:, 3:] + torch.cross(cvel[:, :3], offset, dim=-1)
            return torch.cat([cvel[:, :3], lin], dim=-1)

        elif ot == ObservationType.BODY_VEL:
            body_id = self._model.body(name).id
            root_id = self._model.body_rootid[body_id]
            cvel = wp.to_torch(self._data_wp.cvel)[:, body_id, :]
            xpos = wp.to_torch(self._data_wp.xpos)[:, body_id, :]
            subtree_com = wp.to_torch(self._data_wp.subtree_com)[:, root_id, :]
            offset = xpos - subtree_com
            ang = cvel[:, :3]
            lin = cvel[:, 3:] + torch.cross(ang, offset, dim=-1)
            R = wp.to_torch(self._data_wp.xmat)[:, body_id, :, :]
            Rt = R.transpose(-2, -1)
            return torch.cat(
                [
                    torch.einsum("nij,nj->ni", Rt, ang),
                    torch.einsum("nij,nj->ni", Rt, lin),
                ],
                dim=-1,
            )
        elif ot == ObservationType.JOINT_POS:
            jnt = self._model.joint(name)
            adr = self._model.jnt_qposadr[jnt.id]
            size = ObservationHelper._obs_size(self._model, name, ot)
            return wp.to_torch(self._data_wp.qpos)[:, adr : adr + size]
        elif ot == ObservationType.JOINT_VEL:
            jnt = self._model.joint(name)
            adr = self._model.jnt_dofadr[jnt.id]
            size = ObservationHelper._obs_size(self._model, name, ot)
            return wp.to_torch(self._data_wp.qvel)[:, adr : adr + size]
        elif ot == ObservationType.SITE_POS:
            return wp.to_torch(self._data_wp.site_xpos)[:, self._model.site(name).id, :]
        elif ot == ObservationType.SITE_ROT:
            mat = wp.to_torch(self._data_wp.site_xmat)[
                :, self._model.site(name).id, :, :
            ]
            return mat.reshape(mat.shape[0], 9)
        else:
            raise ValueError(f"Unsupported observation type for _read_warp_field: {ot}")

    # ------------------------------------------------------------------
    # Collision detection — left as numpy for now.
    # Contacts are variable-length and small; vectorizing to torch is complex
    # and the win is minor. Refactor later if needed.
    # ------------------------------------------------------------------

    def _check_collision(self, group1, group2):
        ids1 = self.collision_groups[group1]
        ids2 = self.collision_groups[group2]
        device = TorchUtils.get_device()

        ncon = int(wp.to_torch(self._data_wp.ncollision)[0].item())
        if ncon == 0:
            return torch.zeros(self._num_envs, dtype=torch.bool, device=device)

        geom = wp.to_torch(self._data_wp.contact.geom)[:ncon]  # (ncon, 2)
        worldid = wp.to_torch(self._data_wp.contact.worldid)[:ncon]  # (ncon,)

        # Build boolean lookup tensors: which geom ids belong to each group
        ngeoms = self._model.ngeom
        ids1_mask = torch.zeros(ngeoms, dtype=torch.bool, device=device)
        ids2_mask = torch.zeros(ngeoms, dtype=torch.bool, device=device)
        ids1_mask[list(ids1)] = True
        ids2_mask[list(ids2)] = True

        g1 = geom[:, 0].long()
        g2 = geom[:, 1].long()

        # Match either direction
        match = (ids1_mask[g1] & ids2_mask[g2]) | (
            ids1_mask[g2] & ids2_mask[g1]
        )  # (ncon,)

        # Scatter into per-env result: any match in world_id sets that env to True
        result = torch.zeros(self._num_envs, dtype=torch.bool, device=device)
        if match.any():
            matched_envs = worldid[match].long()
            result[matched_envs] = True
        return result

    def _get_collision_force(self, group1, group2):
        ids1 = self.collision_groups[group1]
        ids2 = self.collision_groups[group2]
        device = TorchUtils.get_device()

        ncon = int(wp.to_torch(self._data_wp.ncollision)[0].item())
        result = torch.zeros((self._num_envs, 6), dtype=torch.float64, device=device)
        if ncon == 0:
            return result

        geom = wp.to_torch(self._data_wp.contact.geom)[:ncon]  # (ncon, 2)
        worldid = wp.to_torch(self._data_wp.contact.worldid)[:ncon]  # (ncon,)
        frame = wp.to_torch(self._data_wp.contact.frame)[:ncon]  # (ncon, ...)

        ngeoms = self._model.ngeom
        ids1_mask = torch.zeros(ngeoms, dtype=torch.bool, device=device)
        ids2_mask = torch.zeros(ngeoms, dtype=torch.bool, device=device)
        ids1_mask[list(ids1)] = True
        ids2_mask[list(ids2)] = True

        g1 = geom[:, 0].long()
        g2 = geom[:, 1].long()
        match = (ids1_mask[g1] & ids2_mask[g2]) | (ids1_mask[g2] & ids2_mask[g1])

        if match.any():
            matched_worldids = worldid[match].long()
            matched_frames = frame[match, :6].to(result.dtype)
            # scatter: overwrites; if multiple matches per env, last one wins (matches original behavior)
            result[matched_worldids] = matched_frames

        return result

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def get_all_observation_keys(self):
        return self.obs_helper.get_all_observation_keys()

    @property
    def dt(self):
        return self._timestep * self._n_intermediate_steps * self._n_substeps
