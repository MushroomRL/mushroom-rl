import numpy as np
import pybullet
from .observation import PyBulletObservationType
from .contacts import ContactHelper


class IndexMap(object):
    def __init__(self, client, model_map, actuation_spec, observation_spec):
        self._client = client
        self.model_map = model_map
        self.joint_map = dict()
        self.link_map = dict()

        self._build_joint_and_link_maps()

        # Contact utils
        contact_types = [PyBulletObservationType.CONTACT_FLAG]
        contacts = [obs[0] for obs in observation_spec if obs[1] in contact_types]
        self._contacts = ContactHelper(client, contacts, self.model_map, self.link_map)

        # Read the actuation spec and build the mapping between actions and ids as well as their limits
        self.action_data = list()
        self._action_low, self._action_high = self._process_actuation_spec(actuation_spec)

        # Read the observation spec to build a mapping at every step.
        # It is ensured that the values appear in the order they are specified.
        self.observation_map = observation_spec
        self.observation_indices_map = dict()

        # We can only specify limits for the joints, all other information can be potentially unbounded
        self._observation_low, self._observation_high = self._process_observations()

    def create_sim_state(self):
        data_obs = list()

        self._contacts.compute_contacts()

        for name, obs_type in self.observation_map:
            if obs_type is PyBulletObservationType.BODY_POS \
               or obs_type is PyBulletObservationType.BODY_LIN_VEL \
               or obs_type is PyBulletObservationType.BODY_ANG_VEL:
                model_id = self.model_map[name]
                if obs_type is PyBulletObservationType.BODY_POS:
                    t, q = self._client.getBasePositionAndOrientation(model_id)
                    data_obs += t + q
                else:
                    v, w = self._client.getBaseVelocity(model_id)
                    if obs_type is PyBulletObservationType.BODY_LIN_VEL:
                        data_obs += v
                    else:
                        data_obs += w
            elif obs_type is PyBulletObservationType.LINK_POS \
                    or obs_type is PyBulletObservationType.LINK_LIN_VEL \
                    or obs_type is PyBulletObservationType.LINK_ANG_VEL:
                model_id, link_id = self.link_map[name]

                if obs_type is PyBulletObservationType.LINK_POS:
                    link_data = self._client.getLinkState(model_id, link_id)
                    t = link_data[0]
                    q = link_data[1]
                    data_obs += t + q
                elif obs_type is PyBulletObservationType.LINK_LIN_VEL:
                    data_obs += self._client.getLinkState(model_id, link_id, computeLinkVelocity=True)[-2]
                elif obs_type is PyBulletObservationType.LINK_ANG_VEL:
                    data_obs += self._client.getLinkState(model_id, link_id, computeLinkVelocity=True)[-1]
            elif obs_type is PyBulletObservationType.JOINT_POS \
                    or obs_type is PyBulletObservationType.JOINT_VEL:
                model_id, joint_id = self.joint_map[name]
                pos, vel, _, _ = self._client.getJointState(model_id, joint_id)
                if obs_type is PyBulletObservationType.JOINT_POS:
                    data_obs.append(pos)
                elif obs_type is PyBulletObservationType.JOINT_VEL:
                    data_obs.append(vel)
            elif obs_type is PyBulletObservationType.CONTACT_FLAG:
                contact = self._contacts.get_contact(name)
                contact_flag = 0 if contact is None else 1
                data_obs.append(contact_flag)

        return np.array(data_obs)

    def apply_control(self, action):

        i = 0
        for model_id, joint_id, mode in self.action_data:
            u = action[i]
            if mode is pybullet.POSITION_CONTROL:
                kwargs = dict(targetPosition=u, maxVelocity=self._client.getJointInfo(model_id, joint_id)[11],
                              force=self._client.getJointInfo(model_id, joint_id)[10])
            elif mode is pybullet.VELOCITY_CONTROL:
                kwargs = dict(targetVelocity=u, maxVelocity=self._client.getJointInfo(model_id, joint_id)[11],
                              force=self._client.getJointInfo(model_id, joint_id)[10])
            elif mode is pybullet.TORQUE_CONTROL:
                kwargs = dict(force=u)
            else:
                raise NotImplementedError

            self._client.setJointMotorControl2(model_id, joint_id, mode, **kwargs)
            i += 1

    def get_index(self, name, obs_type):
        return self.observation_indices_map[name][obs_type]

    def _build_joint_and_link_maps(self):
        for model_id in self.model_map.values():
            for joint_id in range(self._client.getNumJoints(model_id)):
                joint_data = self._client.getJointInfo(model_id, joint_id)

                if joint_data[2] != pybullet.JOINT_FIXED:
                    joint_name = joint_data[1].decode('UTF-8')
                    self.joint_map[joint_name] = (model_id, joint_id)
                link_name = joint_data[12].decode('UTF-8')
                self.link_map[link_name] = (model_id, joint_id)

    def _process_actuation_spec(self, actuation_spec):
        for name, mode in actuation_spec:
            if name in self.joint_map:
                data = self.joint_map[name] + (mode,)
                self.action_data.append(data)

        low = list()
        high = list()

        for model_id, joint_id, mode in self.action_data:
            joint_info = self._client.getJointInfo(model_id, joint_id)
            if mode is pybullet.POSITION_CONTROL:
                low.append(joint_info[8])
                high.append(joint_info[9])
            elif mode is pybullet.VELOCITY_CONTROL:
                low.append(-joint_info[11])
                high.append(joint_info[11])
            elif mode is pybullet.TORQUE_CONTROL:
                low.append(-joint_info[10])
                high.append(joint_info[10])
            else:
                raise NotImplementedError

        return np.array(low), np.array(high)

    def _process_observations(self):
        low = list()
        high = list()

        for name, obs_type in self.observation_map:
            index_count = len(low)
            if obs_type is PyBulletObservationType.BODY_POS \
               or obs_type is PyBulletObservationType.BODY_LIN_VEL \
               or obs_type is PyBulletObservationType.BODY_ANG_VEL:
                n_dim = 7 if obs_type is PyBulletObservationType.BODY_POS else 3
                low += [-np.inf] * n_dim
                high += [np.inf] * n_dim
            elif obs_type is PyBulletObservationType.LINK_POS \
                    or obs_type is PyBulletObservationType.LINK_LIN_VEL \
                    or obs_type is PyBulletObservationType.LINK_ANG_VEL:
                n_dim = 7 if obs_type is PyBulletObservationType.LINK_POS else 3
                low += [-np.inf] * n_dim
                high += [np.inf] * n_dim
            elif obs_type is PyBulletObservationType.JOINT_POS \
                    or obs_type is PyBulletObservationType.JOINT_VEL:
                model_id, joint_id = self.joint_map[name]
                joint_info = self._client.getJointInfo(model_id, joint_id)

                if obs_type is PyBulletObservationType.JOINT_POS:
                    low.append(joint_info[8])
                    high.append(joint_info[9])
                else:
                    max_joint_vel = joint_info[11]
                    low.append(-max_joint_vel)
                    high.append(max_joint_vel)
            elif obs_type is PyBulletObservationType.CONTACT_FLAG:
                low.append(0.)
                high.append(1.)
            else:
                raise RuntimeError('Unsupported observation type')

            self._add_observation_index(name, obs_type, index_count, len(low))

        return np.array(low), np.array(high)

    def _add_observation_index(self, name, obs_type, start, end):
        if name not in self.observation_indices_map:
            self.observation_indices_map[name] = dict()

        self.observation_indices_map[name][obs_type] = list(range(start, end))

    @property
    def observation_limits(self):
        return self._observation_low, self._observation_high

    @property
    def action_limits(self):
        return self._action_low, self._action_high