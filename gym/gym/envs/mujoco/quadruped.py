import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class QuadrupedEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_action = np.zeros(12)
        self.target_vel = np.random.uniform(2, 3)
        #self.target_r_vel = np.random.uniform(-0.2, 0.2)

        mujoco_env.MujocoEnv.__init__(self, 'quadruped.xml', 1)
        utils.EzPickle.__init__(self)

    def step(self, action):
        if np.sum(self.prev_action)==0:
            self.prev_action = action

        self.do_simulation(action, self.frame_skip)
        x_v, y_v, rotat_v = \
            self.data.body_xvelp[1][0], self.data.body_xvelp[1][1], self.data.qvel[1]
        observation = self._get_obs()
        trunk_height = self.data.body_xpos[1][2]

        #r_yaw_v = -1*(rotat_v-self.target_r_vel)**2
        #r_v = -1*(np.sqrt(x_v**2 + y_v**2)-self.target_vel)**2
        foot_id = [1, 2, 3, 4]
        torque_cost = 0.0001*np.sum(np.square(action))
        smoothness_cost = 0.0001*np.sum((self.prev_action-action)**2)
        joint_speed_cost = 0.005*np.sum(np.square(self.data.qvel[:12]))
        b = np.sqrt(x_v**2 + y_v**2)
        lin_vel_base_delta = 4*(np.sqrt(x_v**2 + y_v**2)-self.target_vel)**2
        foot_clearance_cost = 0.1*sum([(self.data.site_xpos[_+1][2]-0.07)**2 for _ in range(4)])
        foot_slip_cost = sum([(self.data.site_xvelp[_+1][2]-0.07)**2 for _ in range(4)])
        reward = -1*(foot_slip_cost+foot_clearance_cost+ joint_speed_cost +
            lin_vel_base_delta+smoothness_cost+torque_cost) + trunk_height * 0.15 + 8

        self.prev_action = action
        done = False
        # if it falls
        if trunk_height < -0.25:
            reward = -200
            done = True
        return observation, reward, done, dict()

    def _get_obs(self):
        sensor_data = np.concatenate(
            [self.data.sensordata, np.concatenate([self.data.body_xpos[1], self.data.qpos], axis=0)], axis=0)
        sensor_data = np.concatenate(
            [sensor_data, np.array([self.target_vel])], axis=0)
        return sensor_data

    def reset_model(self):
        self.prev_action = np.zeros(12)
        self.target_vel = np.random.uniform(2, 3)
        #self.target_r_vel = np.random.uniform(-0.2, 0.2)

        noise_low = -1e-2
        noise_high = 1e-2
        #qpos = self.init_qpos + self.np_random.uniform(
        #    low=noise_low, high=noise_high, size=self.model.nq)
        #qvel = self.init_qvel + self.np_random.uniform(
        #    low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5




