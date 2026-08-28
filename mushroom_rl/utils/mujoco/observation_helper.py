import numpy as np
from enum import Enum

import mujoco


class ObservationType(Enum):
    """
    An enum indicating the type of data that should be added to the observation
    of the environment, can be Joint-/Body-/Site- positions, rotations, and velocities.
    The Observation have the following returns::

        BODY_POS: (3,) x, y, z position of the body
        BODY_ROT: (4,) quaternion of the body
        BODY_VEL: (6,) first angular velocity around x, y, z. Then linear velocity for x, y, z, in local frame
        BODY_VEL_WORLD: (6,) first angular velocity around x, y, z. Then linear velocity for x, y, z, in world frame
        JOINT_POS: (1,) rotation of the joint OR (7,) position, quaternion of a free joint
        JOINT_VEL: (1,) velocity of the joint OR (6,) FIRST linear then angular velocity !different to BODY_VEL!
        SITE_POS: (3,) x, y, z position of the body
        SITE_ROT: (9,) rotation matrix of the site
    """
    __order__ = "BODY_POS BODY_ROT BODY_VEL BODY_VEL_WORLD JOINT_POS JOINT_VEL SITE_POS SITE_ROT"
    BODY_POS = 0
    BODY_ROT = 1
    BODY_VEL = 2
    BODY_VEL_WORLD = 3
    JOINT_POS = 4
    JOINT_VEL = 5
    SITE_POS = 6
    SITE_ROT = 7


class ObservationHelper:
    def __init__(self, observation_spec, model, data, max_joint_velocity, is_warp=False):
        if len(observation_spec) == 0:
            raise AttributeError("No Environment observations were specified. "
                                 "Add at least one observation to the observation_spec.")

        self.obs_low = []
        self.obs_high = []
        self.joint_pos_idx = []
        self.joint_vel_idx = []
        self.joint_mujoco_idx = []

        self.obs_idx_map = {}

        self.build_omit_idx = {}

        self.observation_spec = observation_spec

        if max_joint_velocity is not None:
            max_joint_velocity = iter(max_joint_velocity)

        current_idx = 0
        for key, name, ot in observation_spec:
            assert key not in self.obs_idx_map.keys(), "Found duplicate key in observation specification: \"%s\"" % key
            obs_count = len(self.get_state(model, data, name, ot))
            self.obs_idx_map[key] = list(range(current_idx, current_idx + obs_count))
            self.build_omit_idx[key] = []
            if obs_count == 1 and ot == ObservationType.JOINT_POS:
                self.joint_pos_idx.append(current_idx)
                self.joint_mujoco_idx.append(model.joint(name).id)
                if model.joint(name).limited:
                    self.obs_low.append(model.joint(name).range[0])
                    self.obs_high.append(model.joint(name).range[1])
                else:
                    self.obs_low.append(-np.inf)
                    self.obs_high.append(np.inf)

            elif obs_count == 1 and ot == ObservationType.JOINT_VEL:
                self.joint_vel_idx.append(current_idx)
                if max_joint_velocity is None:
                    max_vel = np.inf
                else:
                    max_vel = next(max_joint_velocity)

                self.obs_low.append(-max_vel)
                self.obs_high.append(max_vel)
            else:
                self.obs_low.extend([-np.inf] * obs_count)
                self.obs_high.extend([np.inf] * obs_count)

            current_idx += obs_count

        self.obs_low = np.array(self.obs_low)
        self.obs_high = np.array(self.obs_high)

        # --- MuJoCo Warp support (additive; no effect on the standard path) ---
        self._is_warp = is_warp
        self._precomputed = []
        if is_warp:
            self._precompute_warp_indices(model)

    def remove_obs(self, key, index):
        """
        Remove an index from the observation. Cannot remove a whole observation, to achieve this just move the
        observation to additional data.
        Helpful for example to remove the z-coordinate from positions if it's not needed
        The index is always of the original observation!
        """
        indices = self.obs_idx_map[key]
        adjusted_index = index - len(self.build_omit_idx[key])

        self.obs_low = np.delete(self.obs_low, indices[adjusted_index])
        self.obs_high = np.delete(self.obs_high, indices[adjusted_index])
        cutoff = indices.pop(adjusted_index)

        for obs_list in self.obs_idx_map.values():
            for idx in range(len(obs_list)):
                if obs_list[idx] > cutoff:
                    obs_list[idx] -= 1

        for i in range(len(self.joint_pos_idx)):
            if self.joint_pos_idx[i] > cutoff:
                self.joint_pos_idx[i] -= 1

        for i in range(len(self.joint_vel_idx)):
            if self.joint_vel_idx[i] > cutoff:
                self.joint_vel_idx[i] -= 1

        self.build_omit_idx[key].append(index)

    def add_obs(self, key, length, min_value=-np.inf, max_value=np.inf):
        """
        Adds an observation entry to the handling logic of the Helper. The observation still has to be manually
        appended to the original observation via _create_observation(self, state), but can get be accessed via
        get_from_obs(self, obs, name, o_type) and is in obs_low / obs_high
        """
        self.obs_idx_map[key] = list(range(len(self.obs_low), len(self.obs_low) + length))

        if hasattr(min_value, "__len__"):
            self.obs_low = np.append(self.obs_low, min_value)
        else:
            self.obs_low = np.append(self.obs_low, [min_value] * length)

        if hasattr(max_value, "__len__"):
            self.obs_high = np.append(self.obs_high, max_value)
        else:
            self.obs_high = np.append(self.obs_high, [max_value] * length)

    def get_from_obs(self, obs, key):
        # Cannot use advanced indexing because it returns a copy.....
        # We want this data to be writeable
        # The Ellipsis makes the same slice work for batched (nworld, obs_dim)
        # observations coming from mujoco_warp; for 1-D obs it is equivalent.
        return obs[..., self.obs_idx_map[key][0]:self.obs_idx_map[key][-1] + 1]

    def get_joint_pos_from_obs(self, obs):
        return obs[self.joint_pos_idx]

    def get_joint_vel_from_obs(self, obs):
        return obs[self.joint_vel_idx]

    def get_obs_limits(self):
        return self.obs_low, self.obs_high

    def get_joint_pos_limits(self):
        return self.obs_low[self.joint_pos_idx], self.obs_high[self.joint_pos_idx]

    def get_joint_vel_limits(self):
        return self.obs_low[self.joint_vel_idx], self.obs_high[self.joint_vel_idx]

    def _build_obs(self, model, data):
        """
        Builds the observation given the true state of the simulation. The ObservationType documentation
        describes the different returns in detail
        Args:
            data: The data of the mujoco sim

        Returns: np.array with all the observations defined by observation_spec
        """
        observations = []
        for key, name, o_type in self.observation_spec:
            omit = np.array(self.build_omit_idx[key])
            obs = self.get_state(model, data, name, o_type)
            if len(omit) != 0:
                obs = np.delete(obs, omit)
            observations.append(obs)
        return np.concatenate(observations)

    def _modify_data(self, model, data, obs):
        """
        Write the values of the observation into the provided mujoco data object. ONLY joint_pos / joint_vel
        observations will have an effect on the simulation when overwritten. Everything else is just discarded by mujoco
        """
        current_idx = 0
        for key, name, o_type in self.observation_spec:
            omit = np.array(self.build_omit_idx[key])
            current_obs = self.get_state(model, data, name, o_type)
            for i in range(len(current_obs)):
                if i not in omit:
                    current_obs[i] = obs[current_idx]
                    current_idx += 1

    def get_state(self, model, data, name, o_type):
        """
        Get a single observation from data, given it's name and observation type. The ObservationType documentation
        describes the different returns in detail
        """
        if o_type == ObservationType.BODY_POS:
            obs = data.body(name).xpos
        elif o_type == ObservationType.BODY_ROT:
            obs = data.body(name).xquat
        elif o_type == ObservationType.BODY_VEL or o_type == ObservationType.BODY_VEL_WORLD:
            local = o_type == ObservationType.BODY_VEL
            obs = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, data.body(name).id, obs, local)
        elif o_type == ObservationType.JOINT_POS:
            obs = data.joint(name).qpos
        elif o_type == ObservationType.JOINT_VEL:
            obs = data.joint(name).qvel
        elif o_type == ObservationType.SITE_POS:
            obs = data.site(name).xpos
        elif o_type == ObservationType.SITE_ROT:
            # Sites don't have rotation quaternion for some reason...
            # x_mat is rotation matrix with shape (9,)
            obs = data.site(name).xmat
        else:
            raise ValueError('Invalid observation type')

        return np.atleast_1d(obs)

    def get_all_observation_keys(self):
        return list(self.obs_idx_map.keys())

    # ------------------------------------------------------------------
    # MuJoCo Warp support
    #
    # Everything below is only used when the helper is constructed with
    # is_warp=True by a mujoco_warp batched environment. The standard
    # MuJoCo path above is unaffected.
    # ------------------------------------------------------------------

    @staticmethod
    def _obs_size(model, name, ot):
        """
        Return the number of scalar values for this observation entry,
        computed from the model alone (no MjData required).
        """
        if ot == ObservationType.BODY_POS:
            return 3
        elif ot == ObservationType.BODY_ROT:
            return 4
        elif ot in (ObservationType.BODY_VEL, ObservationType.BODY_VEL_WORLD):
            return 6
        elif ot == ObservationType.JOINT_POS:
            jnt_type = model.jnt_type[model.joint(name).id]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                return 7
            elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                return 4
            else:
                return 1
        elif ot == ObservationType.JOINT_VEL:
            jnt_type = model.jnt_type[model.joint(name).id]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                return 6
            elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                return 3
            else:
                return 1
        elif ot == ObservationType.SITE_POS:
            return 3
        elif ot == ObservationType.SITE_ROT:
            return 9
        else:
            raise ValueError(f'Invalid observation type: {ot}')

    def _precompute_warp_indices(self, model):
        """
        Precompute static model indices so that batched observations can be
        assembled with tensor slicing instead of per-step name lookups.
        """
        self._precomputed = []
        for key, name, ot in self.observation_spec:
            obs_count = len(self.obs_idx_map[key])
            if ot in (ObservationType.BODY_POS, ObservationType.BODY_ROT,
                      ObservationType.BODY_VEL, ObservationType.BODY_VEL_WORLD):
                self._precomputed.append((key, ot, model.body(name).id, 0))
            elif ot == ObservationType.JOINT_POS:
                jnt = model.joint(name)
                self._precomputed.append((key, ot, model.jnt_qposadr[jnt.id], obs_count))
            elif ot == ObservationType.JOINT_VEL:
                jnt = model.joint(name)
                self._precomputed.append((key, ot, model.jnt_dofadr[jnt.id], obs_count))
            elif ot in (ObservationType.SITE_POS, ObservationType.SITE_ROT):
                self._precomputed.append((key, ot, model.site(name).id, 0))

    def build_obs(self, data_wp):
        """
        Build batched observations from a MuJoCo Warp data object. Only
        available when the helper was constructed with is_warp=True; the
        standard MuJoCo path uses _build_obs(model, data) instead.

        Returns:
            torch.Tensor of shape (nworld, obs_dim).
        """
        assert self._is_warp, "build_obs is only available with is_warp=True; " \
                              "use _build_obs(model, data) for standard MuJoCo."
        return self._build_obs_warp(data_wp)

    def _build_obs_warp(self, data_wp):
        import warp as wp
        import torch

        needed = {ot for _, ot, _, _ in self._precomputed}

        xpos      = wp.to_torch(data_wp.xpos)      if ObservationType.BODY_POS in needed else None
        xquat     = wp.to_torch(data_wp.xquat)     if ObservationType.BODY_ROT in needed else None
        needs_vel = ObservationType.BODY_VEL in needed or ObservationType.BODY_VEL_WORLD in needed
        cvel      = wp.to_torch(data_wp.cvel)      if needs_vel else None
        xmat      = wp.to_torch(data_wp.xmat)      if ObservationType.BODY_VEL in needed else None
        qpos      = wp.to_torch(data_wp.qpos)      if ObservationType.JOINT_POS in needed else None
        qvel      = wp.to_torch(data_wp.qvel)      if ObservationType.JOINT_VEL in needed else None
        site_xpos = wp.to_torch(data_wp.site_xpos) if ObservationType.SITE_POS in needed else None
        site_xmat = wp.to_torch(data_wp.site_xmat) if ObservationType.SITE_ROT in needed else None

        obs_chunks = []
        for key, ot, idx1, idx2 in self._precomputed:
            if ot == ObservationType.BODY_POS:
                chunk = xpos[:, idx1, :]                                   # (nworld, 3)
            elif ot == ObservationType.BODY_ROT:
                chunk = xquat[:, idx1, :]                                  # (nworld, 4)
            elif ot == ObservationType.BODY_VEL_WORLD:
                chunk = cvel[:, idx1, :]                                   # (nworld, 6)
            elif ot == ObservationType.BODY_VEL:
                R   = xmat[:, idx1, :, :]                                  # (nworld, 3, 3)
                Rt  = R.transpose(-2, -1)
                vel = cvel[:, idx1, :]                                     # (nworld, 6)
                chunk = torch.cat([
                    torch.einsum('nij,nj->ni', Rt, vel[:, :3]),
                    torch.einsum('nij,nj->ni', Rt, vel[:, 3:]),
                ], dim=-1)                                                  # (nworld, 6)
            elif ot == ObservationType.JOINT_POS:
                chunk = qpos[:, idx1:idx1 + idx2]                          # (nworld, size)
            elif ot == ObservationType.JOINT_VEL:
                chunk = qvel[:, idx1:idx1 + idx2]                          # (nworld, size)
            elif ot == ObservationType.SITE_POS:
                chunk = site_xpos[:, idx1, :]                              # (nworld, 3)
            elif ot == ObservationType.SITE_ROT:
                chunk = site_xmat[:, idx1, :, :].reshape(-1, 9)            # (nworld, 9)
            else:
                raise ValueError(f'Invalid observation type: {ot}')

            omit = np.array(self.build_omit_idx[key])
            if len(omit) != 0:
                keep = [i for i in range(chunk.shape[1]) if i not in omit]
                chunk = chunk[:, keep]
            obs_chunks.append(chunk)

        return torch.cat(obs_chunks, dim=-1)  # (nworld, obs_dim)

    def _modify_warp_data(self, data_wp, obs, env_indices):
        """
        Write joint states from obs into the warp data for the specified
        environments. Uses wp.to_torch views for zero-copy in-place writes
        on the GPU. Only JOINT_POS and JOINT_VEL affect simulation state.
        """
        import warp as wp

        qpos_view = wp.to_torch(data_wp.qpos)
        qvel_view = wp.to_torch(data_wp.qvel)

        current_idx = 0
        for key, ot, idx1, idx2 in self._precomputed:
            if ot == ObservationType.JOINT_POS:
                qpos_view[env_indices, idx1:idx1 + idx2] = obs[:, current_idx:current_idx + idx2]
                current_idx += idx2
            elif ot == ObservationType.JOINT_VEL:
                qvel_view[env_indices, idx1:idx1 + idx2] = obs[:, current_idx:current_idx + idx2]
                current_idx += idx2
            else:
                current_idx += len(self.obs_idx_map[key])
