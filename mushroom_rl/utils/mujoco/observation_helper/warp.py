import numpy as np
import torch
import warp as wp

from .base import ObservationHelper, ObservationType


class WarpObservationHelper(ObservationHelper):
    """
    Observation helper of a MuJoCo Warp simulation, reading from and writing to a batched data structure holding one
    world per environment. Observations are built and written for every world at once, and a world is addressed by its
    index.

    """
    _WARP_FIELDS_BY_TYPE = {
        ObservationType.BODY_POS: ('xpos',),
        ObservationType.BODY_ROT: ('xquat',),
        ObservationType.BODY_VEL_WORLD: ('xpos', 'subtree_com', 'cvel'),
        ObservationType.BODY_VEL: ('xpos', 'subtree_com', 'cvel', 'xmat'),
        ObservationType.JOINT_POS: ('qpos',),
        ObservationType.JOINT_VEL: ('qvel',),
        ObservationType.SITE_POS: ('site_xpos',),
        ObservationType.SITE_ROT: ('site_xmat',),
    }

    def __init__(self, observation_spec, model, data_wp, max_joint_velocity):
        """
        Constructor.

        Args:
            observation_spec (list): the observation specification, as a list of (key, name, ObservationType) tuples;
            model: the MuJoCo model the observations are read from;
            data_wp: the batched data of the mujoco_warp sim the observations are read from and written to;
            max_joint_velocity (list, None): the maximum velocity of every JOINT_VEL entry of the specification, in
                the order the entries appear, or None to leave them unbounded.

        """
        super().__init__(observation_spec, model, max_joint_velocity)

        self._data_wp = data_wp
        self._precomputed = []
        self._precompute_warp_indices()

    def build_obs(self):
        """
        Build the batched observations from the current state of the simulation.

        Returns:
            A torch.Tensor of shape (num_envs, obs_dim).

        """
        raw = self._load_warp_fields()
        builders = {
            ObservationType.BODY_POS: self._warp_chunk_body_pos,
            ObservationType.BODY_ROT: self._warp_chunk_body_rot,
            ObservationType.BODY_VEL_WORLD: self._warp_chunk_body_vel_world,
            ObservationType.BODY_VEL: self._warp_chunk_body_vel,
            ObservationType.JOINT_POS: self._warp_chunk_joint_pos,
            ObservationType.JOINT_VEL: self._warp_chunk_joint_vel,
            ObservationType.SITE_POS: self._warp_chunk_site_pos,
            ObservationType.SITE_ROT: self._warp_chunk_site_rot,
        }

        obs_chunks = []
        for key, ot, idx1, idx2 in self._precomputed:
            if ot not in builders:
                raise ValueError(f"Invalid observation type: {ot}")
            chunk = builders[ot](raw, idx1, idx2)

            omit = np.array(self.build_omit_idx[key])
            if len(omit) != 0:
                keep = [i for i in range(chunk.shape[1]) if i not in omit]
                chunk = chunk[:, keep]
            obs_chunks.append(chunk)

        return torch.cat(obs_chunks, dim=-1)

    def modify_data(self, obs, env_indices=None):
        """
        Write the values of the observation into the batched data, for the worlds listed in env_indices. Only
        joint_pos / joint_vel observations will have an effect on the simulation when overwritten. Everything else is
        just discarded by mujoco.

        Args:
            obs (torch.Tensor): observations of every world, shape (num_envs, obs_dim);
            env_indices (torch.Tensor, None): indices of the worlds to update, or None to update all of them.

        """
        qpos_view = wp.to_torch(self._data_wp.qpos)
        qvel_view = wp.to_torch(self._data_wp.qvel)

        if env_indices is None:
            env_indices = torch.arange(qpos_view.shape[0], device=qpos_view.device, dtype=torch.long)

        obs = obs[env_indices]

        views = {ObservationType.JOINT_POS: qpos_view, ObservationType.JOINT_VEL: qvel_view}

        current_idx = 0
        for key, ot, idx1, idx2 in self._precomputed:
            if ot in views:
                omit = self.build_omit_idx[key]
                if omit:
                    keep = [i for i in range(idx2) if i not in omit]
                    columns = idx1 + torch.tensor(keep, device=views[ot].device, dtype=torch.long)
                    views[ot][env_indices.unsqueeze(1), columns.unsqueeze(0)] = obs[
                        :, current_idx:current_idx + len(keep)
                    ]
                    current_idx += len(keep)
                else:
                    views[ot][env_indices, idx1:idx1 + idx2] = obs[
                        :, current_idx:current_idx + idx2
                    ]
                    current_idx += idx2
            else:
                current_idx += len(self.obs_idx_map[key])

    def get_state(self, name, o_type, env_indices=None):
        """
        Read a single entry from the batched data, given its name and observation type.

        Args:
            name (str): the name of the object in the XML specification;
            o_type (ObservationType): the type of data to read;
            env_indices (torch.Tensor, None): indices of the worlds to read, or None to read all of them.

        Returns:
            The requested data, as a torch.Tensor of shape (len(env_indices) or num_envs, ...).

        """
        if o_type == ObservationType.BODY_POS:
            data = wp.to_torch(self._data_wp.xpos)[:, self._address(name, o_type), :]
        elif o_type == ObservationType.BODY_ROT:
            data = wp.to_torch(self._data_wp.xquat)[:, self._address(name, o_type), :]
        elif o_type == ObservationType.BODY_VEL_WORLD:
            body_id = self._address(name, o_type)
            root_id = self._model.body_rootid[body_id]
            cvel = wp.to_torch(self._data_wp.cvel)[:, body_id, :]
            xpos = wp.to_torch(self._data_wp.xpos)[:, body_id, :]
            subtree_com = wp.to_torch(self._data_wp.subtree_com)[:, root_id, :]
            offset = xpos - subtree_com
            lin = cvel[:, 3:] + torch.cross(cvel[:, :3], offset, dim=-1)
            data = torch.cat([cvel[:, :3], lin], dim=-1)
        elif o_type == ObservationType.BODY_VEL:
            body_id = self._address(name, o_type)
            root_id = self._model.body_rootid[body_id]
            cvel = wp.to_torch(self._data_wp.cvel)[:, body_id, :]
            xpos = wp.to_torch(self._data_wp.xpos)[:, body_id, :]
            subtree_com = wp.to_torch(self._data_wp.subtree_com)[:, root_id, :]
            offset = xpos - subtree_com
            ang = cvel[:, :3]
            lin = cvel[:, 3:] + torch.cross(ang, offset, dim=-1)
            Rt = wp.to_torch(self._data_wp.xmat)[:, body_id, :, :].transpose(-2, -1)
            data = torch.cat(
                [
                    torch.einsum("nij,nj->ni", Rt, ang),
                    torch.einsum("nij,nj->ni", Rt, lin),
                ],
                dim=-1,
            )
        elif o_type == ObservationType.JOINT_POS:
            data = wp.to_torch(self._data_wp.qpos)[:, self._address(name, o_type)]
        elif o_type == ObservationType.JOINT_VEL:
            data = wp.to_torch(self._data_wp.qvel)[:, self._address(name, o_type)]
        elif o_type == ObservationType.SITE_POS:
            data = wp.to_torch(self._data_wp.site_xpos)[:, self._address(name, o_type), :]
        elif o_type == ObservationType.SITE_ROT:
            mat = wp.to_torch(self._data_wp.site_xmat)[:, self._address(name, o_type), :, :]
            data = mat.reshape(mat.shape[0], 9)
        else:
            raise ValueError("Invalid observation type")

        return self._select_worlds(data, env_indices)

    def set_state(self, name, o_type, value, env_indices=None):
        """
        Write a single entry into the batched data, given its name and observation type. Only JOINT_POS and JOINT_VEL
        entries are supported.

        Args:
            name (str): the name of the object in the XML specification;
            o_type (ObservationType): the type of data to write;
            value (torch.Tensor): the data to write, shape (len(env_indices) or num_envs, ...);
            env_indices (torch.Tensor, None): indices of the worlds to write to, or None to write to all of them.

        """
        if o_type == ObservationType.JOINT_POS:
            view = wp.to_torch(self._data_wp.qpos)
        elif o_type == ObservationType.JOINT_VEL:
            view = wp.to_torch(self._data_wp.qvel)
        else:
            raise ValueError(f"set_state only supports JOINT_POS and JOINT_VEL; got {o_type}.")

        device = view.device

        if env_indices is None:
            env_indices = torch.arange(view.shape[0], device=device, dtype=torch.long)

        value_t = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=device)

        address = self._address(name, o_type)
        col_idx = torch.arange(address.start, address.stop, device=device, dtype=torch.long)
        view[env_indices.unsqueeze(1), col_idx.unsqueeze(0)] = value_t.to(view.dtype)

    def _select_worlds(self, data, env_indices):
        """
        Args:
            data (torch.Tensor): data of every world, shape (num_envs, ...);
            env_indices (torch.Tensor, None): indices of the worlds to keep, or None to keep all of them.

        Returns:
            The rows of the given data selected by env_indices.

        """
        if env_indices is not None:
            if not isinstance(env_indices, torch.Tensor):
                env_indices = torch.as_tensor(
                    env_indices, device=data.device, dtype=torch.long
                )
            return data[env_indices]

        return data

    def _precompute_warp_indices(self):
        """
        Precompute the static model indices used to assemble batched observations by tensor slicing.
        """
        self._precomputed = []
        for key, name, ot in self.observation_spec:
            obs_count = len(self.obs_idx_map[key])
            if ot in (
                ObservationType.BODY_POS,
                ObservationType.BODY_ROT,
                ObservationType.BODY_VEL,
                ObservationType.BODY_VEL_WORLD,
            ):
                body_id = self._model.body(name).id
                root_id = self._model.body_rootid[body_id]
                self._precomputed.append((key, ot, body_id, root_id))
            elif ot == ObservationType.JOINT_POS:
                jnt = self._model.joint(name)
                self._precomputed.append(
                    (key, ot, self._model.jnt_qposadr[jnt.id], obs_count)
                )
            elif ot == ObservationType.JOINT_VEL:
                jnt = self._model.joint(name)
                self._precomputed.append((key, ot, self._model.jnt_dofadr[jnt.id], obs_count))
            elif ot in (ObservationType.SITE_POS, ObservationType.SITE_ROT):
                self._precomputed.append((key, ot, self._model.site(name).id, 0))

    def _load_warp_fields(self):
        """
        Load, as torch tensors, only the raw mujoco_warp data fields required by the observation types actually
        present in _precomputed.

        Returns:
            A dictionary mapping field name to its torch tensor.
        """
        needed = {ot for _, ot, _, _ in self._precomputed}
        field_names = set()
        for ot in needed:
            field_names.update(self._WARP_FIELDS_BY_TYPE[ot])

        return {name: wp.to_torch(getattr(self._data_wp, name)) for name in field_names}

    @staticmethod
    def _warp_chunk_body_pos(raw, idx1, _):
        return raw['xpos'][:, idx1, :]

    @staticmethod
    def _warp_chunk_body_rot(raw, idx1, _):
        return raw['xquat'][:, idx1, :]

    @staticmethod
    def _warp_chunk_body_vel_world(raw, idx1, idx2):
        vel = raw['cvel'][:, idx1, :]
        offset = raw['xpos'][:, idx1, :] - raw['subtree_com'][:, idx2, :]
        ang = vel[:, :3]
        lin = vel[:, 3:] + torch.cross(ang, offset, dim=-1)
        return torch.cat([ang, lin], dim=-1)

    @staticmethod
    def _warp_chunk_body_vel(raw, idx1, idx2):
        vel = raw['cvel'][:, idx1, :]
        offset = raw['xpos'][:, idx1, :] - raw['subtree_com'][:, idx2, :]
        ang = vel[:, :3]
        lin = vel[:, 3:] + torch.cross(ang, offset, dim=-1)
        Rt = raw['xmat'][:, idx1, :, :].transpose(-2, -1)
        return torch.cat(
            [
                torch.einsum("nij,nj->ni", Rt, ang),
                torch.einsum("nij,nj->ni", Rt, lin),
            ],
            dim=-1,
        )

    @staticmethod
    def _warp_chunk_joint_pos(raw, idx1, idx2):
        return raw['qpos'][:, idx1:idx1 + idx2]

    @staticmethod
    def _warp_chunk_joint_vel(raw, idx1, idx2):
        return raw['qvel'][:, idx1:idx1 + idx2]

    @staticmethod
    def _warp_chunk_site_pos(raw, idx1, _):
        return raw['site_xpos'][:, idx1, :]

    @staticmethod
    def _warp_chunk_site_rot(raw, idx1, _):
        return raw['site_xmat'][:, idx1, :, :].reshape(-1, 9)
