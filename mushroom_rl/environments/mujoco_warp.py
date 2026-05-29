import mujoco
import mujoco_warp as mj_warp
import torch
import numpy as np
import warp as wp
from dm_control import mjcf

from mushroom_rl.core import VectorizedEnvironment, MDPInfo, ArrayBackend
from mushroom_rl.rl_utils.spaces import Box
from mushroom_rl.utils.mujoco import ObservationHelper, ObservationType, MujocoViewer
from mushroom_rl.utils.torch import TorchUtils
from mushroom_rl.environments.mujoco import MuJoCo


class MuJoCoWarp(VectorizedEnvironment):
    """
    Class to create N parallel environments using MuJoCo Warp for GPU-accelerated batch simulation.

    """

    def __init__(self, xml_file, actuation_spec, observation_spec, gamma, horizon, num_envs,
                 timestep=None, n_substeps=1, n_intermediate_steps=1, additional_data_spec=None,
                 collision_groups=None, max_joint_vel=None, nconmax=None, njmax=None,
                 **viewer_params):
        """
        Constructor.

        Args:
            xml_file (str/xml handle): A path to the MuJoCo XML file or a dm_control mjcf handle.
            actuation_spec (list): Names of the actuators controllable by the agent. Pass an empty
                list to use all actuators.
            observation_spec (list): List of (key, name, ObservationType) tuples describing the
                observation.
            gamma (float): Discount factor.
            horizon (int): Maximum episode horizon.
            num_envs (int): Number of parallel worlds to simulate.
            timestep (float, None): Physics timestep. Uses the XML default when None.
            n_substeps (int): Number of MuJoCo substeps per control step.
            n_intermediate_steps (int): Number of control steps per agent action.
            additional_data_spec (list, None): List of (key, name, ObservationType) tuples for
                data that should be readable/writable but not included in the observation.
            collision_groups (list, None): List of (key, geom_names) pairs for collision checking.
            max_joint_vel (list, None): Maximum joint velocities for JOINT_VEL observations.
            nconmax (int, None): Max contacts per world (None = mujoco_warp default).
            njmax (int, None): Max constraints per world (None = mujoco_warp default).
            **viewer_params: Extra keyword arguments forwarded to MujocoViewer.
        """
        self._mj_warp = mj_warp
        self._wp = wp

        # Load model
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

        # Build warp model and batched data
        self._model_wp = mj_warp.put_model(self._model)
        self._data_wp  = mj_warp.make_data(
            self._model, nworld=num_envs, nconmax=nconmax, njmax=njmax
        )

        # Resolve action indices using a temporary CPU data object
        _tmp_data = mujoco.MjData(self._model)
        self._action_indices = MuJoCo.get_action_indices(self._model, _tmp_data, actuation_spec)
        action_space = MuJoCo.get_action_space(self._action_indices, self._model)

        # Observation helper
        self.obs_helper = ObservationHelper(
            observation_spec, self._model, is_warp=True, max_joint_velocity=max_joint_vel
        )
        observation_space = Box(*self.obs_helper.get_obs_limits())

        # Additional data
        self.additional_data = {}
        if additional_data_spec is not None:
            for key, name, ot in additional_data_spec:
                self.additional_data[key] = (name, ot)

        # Collision groups
        self.collision_groups = {}
        if collision_groups is not None:
            for name, geom_names in collision_groups:
                col_group = []
                for geom_name in geom_names:
                    mj_id = mujoco.mj_name2id(
                        self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
                    )
                    assert mj_id != -1, \
                        f'geom "{geom_name}" not found! Can\'t be used for collision-checking.'
                    col_group.append(mj_id)
                self.collision_groups[name] = set(col_group)

        mdp_info = MDPInfo(observation_space, action_space, gamma, horizon, self.dt, backend='torch')
        mdp_info = self._modify_mdp_info(mdp_info)

        mujoco.set_mju_user_warning(MuJoCo.user_warning_raise_exception)

        self._recompute_action_per_step = (
            type(self)._compute_action != MuJoCoWarp._compute_action
        )

        super().__init__(mdp_info, num_envs)

    # ------------------------------------------------------------------
    # VectorizedEnvironment interface
    # ------------------------------------------------------------------

    def step_all(self, env_mask, action):
        """
        Perform one agent step for all active environments.

        Args:
            env_mask (np.ndarray): Boolean array of shape (num_envs,) marking active envs.
            action (np.ndarray): Actions of shape (num_envs, n_actions).

        Returns:
            cur_obs (np.ndarray): Observations of shape (num_envs, obs_dim).
            reward (np.ndarray): Rewards of shape (num_envs,).
            absorbing (np.ndarray): Boolean array of shape (num_envs,).
            info (dict): Extra info dictionary.
        """
        action = self._preprocess_action(action)

        self._step_init(self._obs, action)

        ctrl_action = None

        for i in range(self._n_intermediate_steps):
            if self._recompute_action_per_step or ctrl_action is None:
                ctrl_action = self._compute_action(self._obs, action)
                self._set_ctrl(ctrl_action, env_mask)

            self._simulation_pre_step()

            for _ in range(self._n_substeps):
                self._mj_warp.step(self._model_wp, self._data_wp)

            self._simulation_post_step()

        cur_obs = self._create_observation(self.obs_helper.build_obs(self._data_wp))

        self._step_finalize()

        absorbing = self.is_absorbing(cur_obs)
        reward = self.reward(self._obs, action, cur_obs, absorbing)
        info = self._create_info_dictionary(cur_obs)

        self._obs = cur_obs.clone()

        out_device = TorchUtils.get_device()
        env_mask_t = torch.as_tensor(env_mask, device=cur_obs.device)
        return (
            self._modify_observation(cur_obs).to(out_device),
            reward.to(out_device),
            (absorbing & env_mask_t).to(out_device),
            info,
        )

    def reset_all(self, env_mask, state=None):
        """
        Reset the specified environments to a fresh state.

        Args:
            env_mask (np.ndarray): Boolean array of shape (num_envs,) marking envs to reset.
            state (np.ndarray, None): Optional initial state. Only joint positions/velocities
                are applied (other state is ignored).

        Returns:
            obs (np.ndarray): Observations of shape (num_envs, obs_dim) after reset.
            info (dict): Info dictionary.
        """
        env_indices = np.where(env_mask)[0]

        # reset_data expects a warp bool array of shape (nworld,)
        reset_mask = self._wp.zeros(self._num_envs, dtype=self._wp.bool, device='cuda:0')
        reset_np = reset_mask.numpy()
        reset_np[env_indices] = True
        reset_mask.assign(reset_np)
        self._mj_warp.reset_data(self._model_wp, self._data_wp, reset=reset_mask)

        self.setup(env_indices, state)

        obs = self._create_observation(self.obs_helper.build_obs(self._data_wp))
        obs = self._modify_observation(obs)

        if self._obs is None:
            self._obs = obs.clone()
        else:
            self._obs[env_mask] = obs[env_mask]

        info = self._create_info_dictionary(obs)
        return obs.clone(), info

    def render_all(self, env_mask, record=False):
        """
        Render the default environment using get_data_into for efficient state transfer.

        Args:
            record (bool): If True, return the rendered frame as an np.ndarray.

        Returns:
            Frame as np.ndarray if record=True, else None.
        """
        if self._viewer is None:
            self._viewer = MujocoViewer(
                self._model, self.dt, record=record, **self._viewer_params
            )

        # get_data_into copies the full state of one world into a CPU MjData
        render_data = mujoco.MjData(self._model)
        self._mj_warp.get_data_into(render_data, self._model, self._data_wp,
                                     world_id=self._default_env)

        return self._viewer.render(render_data, record)

    def stop(self):
        if self._viewer is not None:
            self._viewer.stop()
            del self._viewer
            self._viewer = None

    def seed(self, seed):
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Abstract methods – must be overridden by subclasses
    # ------------------------------------------------------------------

    def reward(self, obs, action, next_obs, absorbing):
        """
        Compute vectorized rewards.

        Args:
            obs (np.ndarray): Current observations, shape (num_envs, obs_dim).
            action (np.ndarray): Applied actions, shape (num_envs, n_actions).
            next_obs (np.ndarray): Next observations, shape (num_envs, obs_dim).
            absorbing (np.ndarray): Boolean absorbing flags, shape (num_envs,).

        Returns:
            np.ndarray of shape (num_envs,).
        """
        raise NotImplementedError

    def is_absorbing(self, obs):
        """
        Check whether each environment is in an absorbing state.

        Args:
            obs (np.ndarray): Observations, shape (num_envs, obs_dim).

        Returns:
            np.ndarray of bool, shape (num_envs,).
        """
        raise NotImplementedError

    def setup(self, env_indices, obs):
        """
        Called after reset to apply initial state to the specified environments.
        Override to customise per-environment initial conditions.

        Args:
            env_indices (np.ndarray): 1-D integer array of world indices that were reset.
            obs (np.ndarray, None): Optional initial state. When provided, joint positions and
                velocities are written back to the simulation.
        """
        if obs is not None:
            self.obs_helper._modify_data(self._data_wp, obs, env_indices)
        self._mj_warp.forward(self._model_wp, self._data_wp)

    # ------------------------------------------------------------------
    # Overridable hooks (same pattern as MuJoCo)
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

        Args:
            ctrl_action (np.ndarray): shape (num_envs, n_actions).
            env_mask (np.ndarray): Boolean mask, shape (num_envs,).
        """
        ctrl_np = self._data_wp.ctrl.numpy().copy()
        ctrl_np[env_mask, :] = ctrl_action[env_mask, :][:, self._action_indices]
        self._data_wp.ctrl.assign(ctrl_np)

    def _read_data(self, name, env_indices=None):
        """
        Read a named data field from the warp simulation.

        Args:
            name (str): Key defined in additional_data_spec.
            env_indices (array-like, None): Environments to read. Reads all when None.

        Returns:
            np.ndarray of shape (len(env_indices), size) or (num_envs, size).
        """
        field_name, ot = self.additional_data[name]
        data = self._read_warp_field(field_name, ot)
        if env_indices is not None:
            return data[env_indices]
        return data

    def _write_data(self, name, value, env_indices=None):
        """
        Write a named data field into the warp simulation.

        Args:
            name (str): Key defined in additional_data_spec.
            value (np.ndarray): Data to write.
            env_indices (array-like, None): Environments to write. Writes all when None.
        """
        field_name, ot = self.additional_data[name]
        if env_indices is None:
            env_indices = np.arange(self._num_envs)

        if ot == ObservationType.JOINT_POS:
            jnt = self._model.joint(field_name)
            adr = self._model.jnt_qposadr[jnt.id]
            size = ObservationHelper._obs_size(self._model, field_name, ot)
            qpos_np = self._data_wp.qpos.numpy().copy()
            qpos_np[np.ix_(env_indices, list(range(adr, adr + size)))] = value
            self._data_wp.qpos.assign(qpos_np)
        elif ot == ObservationType.JOINT_VEL:
            jnt = self._model.joint(field_name)
            adr = self._model.jnt_dofadr[jnt.id]
            size = ObservationHelper._obs_size(self._model, field_name, ot)
            qvel_np = self._data_wp.qvel.numpy().copy()
            qvel_np[np.ix_(env_indices, list(range(adr, adr + size)))] = value
            self._data_wp.qvel.assign(qvel_np)
        else:
            raise ValueError(
                f'_write_data only supports JOINT_POS and JOINT_VEL; got {ot}.'
            )

    def _read_warp_field(self, name, ot):
        """Return a torch tensor for a given named object and observation type."""
        if ot == ObservationType.BODY_POS:
            return wp.to_torch(self._data_wp.xpos)[:, self._model.body(name).id, :]
        elif ot == ObservationType.BODY_ROT:
            return wp.to_torch(self._data_wp.xquat)[:, self._model.body(name).id, :]
        elif ot == ObservationType.BODY_VEL_WORLD:
            return wp.to_torch(self._data_wp.cvel)[:, self._model.body(name).id, :]
        elif ot == ObservationType.BODY_VEL:
            body_id = self._model.body(name).id
            cvel = wp.to_torch(self._data_wp.cvel)[:, body_id, :]
            R = wp.to_torch(self._data_wp.xmat)[:, body_id, :, :]
            Rt = R.transpose(-2, -1)
            return torch.cat([
                torch.einsum('nij,nj->ni', Rt, cvel[:, :3]),
                torch.einsum('nij,nj->ni', Rt, cvel[:, 3:]),
            ], dim=-1)
        elif ot == ObservationType.JOINT_POS:
            jnt  = self._model.joint(name)
            adr  = self._model.jnt_qposadr[jnt.id]
            size = ObservationHelper._obs_size(self._model, name, ot)
            return wp.to_torch(self._data_wp.qpos)[:, adr:adr + size]
        elif ot == ObservationType.JOINT_VEL:
            jnt  = self._model.joint(name)
            adr  = self._model.jnt_dofadr[jnt.id]
            size = ObservationHelper._obs_size(self._model, name, ot)
            return wp.to_torch(self._data_wp.qvel)[:, adr:adr + size]
        elif ot == ObservationType.SITE_POS:
            return wp.to_torch(self._data_wp.site_xpos)[:, self._model.site(name).id, :]
        elif ot == ObservationType.SITE_ROT:
            mat = wp.to_torch(self._data_wp.site_xmat)[:, self._model.site(name).id, :, :]
            return mat.reshape(mat.shape[0], 9)
        else:
            raise ValueError(f'Unsupported observation type for _read_warp_field: {ot}')

    # ------------------------------------------------------------------
    # Collision detection
    #
    # In mujoco_warp contacts are stored as a flat list across all worlds:
    #   contact.geom      – (total_ncon, 2)  geom id pairs
    #   contact.worldid   – (total_ncon,)    which world each contact belongs to
    #   ncollision        – warp array [total_ncon]  (total valid contacts)
    # ------------------------------------------------------------------

    def _check_collision(self, group1, group2):
        """
        Check whether a collision occurred between the two geom groups for each environment.

        Args:
            group1 (str): Key from collision_groups.
            group2 (str): Key from collision_groups.

        Returns:
            np.ndarray of bool, shape (num_envs,).
        """
        ids1 = self.collision_groups[group1]
        ids2 = self.collision_groups[group2]

        ncon = int(self._data_wp.ncollision.numpy()[0])
        if ncon == 0:
            return np.zeros(self._num_envs, dtype=bool)

        geom_np    = self._data_wp.contact.geom.numpy()[:ncon]      # (ncon, 2)
        worldid_np = self._data_wp.contact.worldid.numpy()[:ncon]   # (ncon,)

        result = np.zeros(self._num_envs, dtype=bool)
        for con_i in range(ncon):
            env_id = int(worldid_np[con_i])
            if result[env_id]:
                continue
            g1, g2 = int(geom_np[con_i, 0]), int(geom_np[con_i, 1])
            if (g1 in ids1 and g2 in ids2) or (g1 in ids2 and g2 in ids1):
                result[env_id] = True
        return result

    def _get_collision_force(self, group1, group2):
        """
        Return the contact force/torque [3D force + 3D torque] between two geom groups per env.
        Returns a zero vector for environments without a matching contact.

        Args:
            group1 (str): Key from collision_groups.
            group2 (str): Key from collision_groups.

        Returns:
            np.ndarray of shape (num_envs, 6).
        """
        ids1 = self.collision_groups[group1]
        ids2 = self.collision_groups[group2]

        ncon = int(self._data_wp.ncollision.numpy()[0])
        result = np.zeros((self._num_envs, 6), dtype=np.float64)
        if ncon == 0:
            return result

        geom_np    = self._data_wp.contact.geom.numpy()[:ncon]      # (ncon, 2)
        worldid_np = self._data_wp.contact.worldid.numpy()[:ncon]   # (ncon,)
        frame_np   = self._data_wp.contact.frame.numpy()[:ncon]     # (ncon, ...)

        for con_i in range(ncon):
            env_id = int(worldid_np[con_i])
            g1, g2 = int(geom_np[con_i, 0]), int(geom_np[con_i, 1])
            if (g1 in ids1 and g2 in ids2) or (g1 in ids2 and g2 in ids1):
                result[env_id] = frame_np[con_i, :6]
        return result

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def get_all_observation_keys(self):
        return self.obs_helper.get_all_observation_keys()

    @property
    def dt(self):
        return self._timestep * self._n_intermediate_steps * self._n_substeps