import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
import os

ADD_BONUS_REWARDS = True
USE_SPARSE_REWARDS = True

class HMSDoor(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, use_image_obs=False, height=100, width=100, camera_id=0, frame_skip=5):
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0
        self.use_image_obs = use_image_obs
        self.height = height
        self.width = width
        self.depth = False
        self.camera_id = camera_id
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_door.xml', frame_skip=frame_skip)

        # change actuator sensitivity
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.sim.model.actuator_gainprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_WRJ1'):self.sim.model.actuator_name2id('A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.sim.model.actuator_biasprm[self.sim.model.actuator_name2id('A_FFJ3'):self.sim.model.actuator_name2id('A_THJ0')+1,:3] = np.array([0, -1, 0])

        utils.EzPickle.__init__(self)
        ob = self.reset_model()
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
        self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])
        self.door_hinge_did = self.model.jnt_dofadr[self.model.joint_name2id('door_hinge')]
        self.grasp_sid = self.model.site_name2id('S_grasp')
        self.handle_sid = self.model.site_name2id('S_handle')
        self.door_bid = self.model.body_name2id('frame')

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a*self.act_rng # mean center and scale
        except:
            a = a                             # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self.get_obs()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = self.data.qpos[self.door_hinge_did]

        # get to handle
        reward = -0.1*np.linalg.norm(palm_pos-handle_pos)
        # open door
        reward += -0.1*(door_pos - 1.57)*(door_pos - 1.57)
        # velocity cost
        reward += -1e-5*np.sum(self.data.qvel**2)

        if ADD_BONUS_REWARDS:
            # Bonus
            if door_pos > 0.2:
                reward += 2
            if door_pos > 1.0:
                reward += 8
            if door_pos > 1.35:
                reward += 10

        if USE_SPARSE_REWARDS:
            reward = -0.1*(door_pos - 1.57)*(door_pos - 1.57)
            if door_pos > 0.2:
                reward += 2
            if door_pos > 1.0:
                reward += 8
            if door_pos > 1.35:
                reward += 10

        goal_achieved = True if door_pos > 1.4 else False

        return ob, reward, False, dict(goal_achieved=goal_achieved)

    def get_obs(self):
        if self.use_image_obs :
            obs = self.render(mode='rgb_array', height=self.height, width=self.width, camera_id=self.camera_id, depth=self.depth)
        else :
            obs = self.render(mode='state')
        return obs

    def render(self, mode='rgb_array', height=100, width=100, camera_id=0, depth=False) :
        if mode == 'rgb_array' : # Hard coded to vil_camera.
            img = self.sim.render(width=width, height=height, camera_name='vil_camera', depth=depth)
            return img[::-1,:,:]
        if mode == 'state' :
            # qpos for hand
            # xpos for obj
            # xpos for target
            qp = self.data.qpos.ravel()
            handle_pos = self.data.site_xpos[self.handle_sid].ravel()
            palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
            door_pos = np.array([self.data.qpos[self.door_hinge_did]])
            if door_pos > 1.0:
                door_open = 1.0
            else:
                door_open = -1.0
            latch_pos = qp[-1]
            return np.concatenate([qp[1:-2], [latch_pos], door_pos, palm_pos, handle_pos, palm_pos-handle_pos, [door_open]])

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.model.body_pos[self.door_bid,0] = self.np_random.uniform(low=-0.3, high=-0.2)
        self.model.body_pos[self.door_bid,1] = self.np_random.uniform(low=0.25, high=0.35)
        self.model.body_pos[self.door_bid,2] = self.np_random.uniform(low=0.252, high=0.35)

        # self.sim.model.light_pos[:] = self.np_random.uniform(low=np.array([-1, -1, 2]), high=np.array([.5, .5, 5]))
        # self.sim.model.light_dir[:] = self.np_random.uniform(low=np.array([-.25, -.25, -1]), high=np.array([.25, .25, -.5]))
        # self.sim.model.light_ambient[:] = self.np_random.uniform(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]))
        # self.model.geom_rgba[52:, :3] = self.np_random.uniform(low=np.array([-.52, -.52, -.52]), high=np.array([.52, .52, .52])) + self.np_random.uniform(.8, 1.2)*self.model.geom_rgba[52:, :3]

        # id_forearm = self.sim.model.geom_name2id('V_forearm')
        # for i in range(self.sim.model.geom_name2id):
            # print(i)
        # self.sim.model.geom_rgba[id_forearm] = (self.np_random.uniform(0., 1.), self.np_random.uniform(0., 1.), self.np_random.uniform(0., 1.), 1)
        # for i in range(1):
        #     self.sim.model.geom_type[i+1] = np.random.randint(4) + 2 # between 2 to 6
        #     self.sim.model.geom_size[i+1] = (self.np_random.uniform(0.03, 0.04), self.np_random.uniform(0.03, 0.04), self.np_random.uniform(0.03, 0.04))
        # #    # self.sim.model.geom_size[i+1] = (0.035, 0.035, 0.035)
        #     self.sim.model.geom_rgba[i+1] = (1, self.np_random.uniform(0., 1.), self.np_random.uniform(0., 1.), 1)
        # #    # self.sim.model.geom_rgba[i+1] = (1, 0, 0, 1)
        #     self.sim.model.geom_pos[i+1] = (self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.2, 0.2), 0.0)

        self.sim.forward()
        return self.get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        door_body_pos = self.model.body_pos[self.door_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)
        self.model.body_pos[self.door_bid] = state_dict['door_body_pos']
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = 90
        self.sim.forward()
        self.viewer.cam.distance = 1.5

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if door open for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
