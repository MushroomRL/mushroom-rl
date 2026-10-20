import numpy as np
import pybullet
from mushroom_rl.utils.pybullet.observation import PyBulletObservationType
from mushroom_rl.utils.pybullet.contacts import ContactHelper


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
            if obs_type.is_contact():
                contact = self._contacts.get_contact(name)
                data_obs.append(0 if contact is None else 1)
            else:
                data = getattr(self._client, obs_type.accessor)(*self._element_ids(name, obs_type),
                                                                **obs_type.accessor_kwargs)
                for component in obs_type.components:
                    if obs_type.is_joint():
                        data_obs.append(data[component])
                    else:
                        data_obs += data[component]

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
            if not isinstance(obs_type, PyBulletObservationType):
                raise RuntimeError('Unsupported observation type')

            index_count = len(low)
            if obs_type.is_joint():
                joint_low, joint_high = self._joint_limits(name, obs_type)
                low.append(joint_low)
                high.append(joint_high)
            elif obs_type.is_contact():
                low.append(0.)
                high.append(1.)
            else:
                low += [-np.inf] * obs_type.length
                high += [np.inf] * obs_type.length

            self._add_observation_index(name, obs_type, index_count, len(low))

        return np.array(low), np.array(high)

    def _joint_limits(self, name, obs_type):
        joint_info = self._client.getJointInfo(*self.joint_map[name])

        if obs_type is PyBulletObservationType.JOINT_POS:
            return joint_info[8], joint_info[9]
        else:
            max_joint_vel = joint_info[11]
            return -max_joint_vel, max_joint_vel

    def _element_ids(self, name, obs_type):
        if obs_type.is_body():
            return self.model_map[name],
        elif obs_type.is_link():
            return self.link_map[name]
        else:
            return self.joint_map[name]

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
