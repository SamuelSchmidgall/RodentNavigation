import numpy as np
from gym import utils
from gym import spaces
from copy import deepcopy
from gym.envs.mujoco import mujoco_env

class HalfCheetahHierEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, layer=1, ll_agent=None):
        self.timestep = 0
        self.layer = layer
        self.layer2_itr = 5
        if self.layer == 1:
            self.target_velocity = np.random.uniform(low=-1.0, high=2.5)
        elif self.layer == 2:
            self.ll_agent = ll_agent
            self.target_delta_x_pos = np.random.uniform(low=-20.0, high=50.0)
            self.original_tdxp = deepcopy(self.target_delta_x_pos)
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 1)
        utils.EzPickle.__init__(self)

        if self.layer == 2:
            self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([2.5]), dtype=np.float32)


    def step(self, action):
        if self.layer == 1:
            x_position_before = self.sim.data.qpos[0]
            self.do_simulation(action, self.frame_skip)
            x_position_after = self.sim.data.qpos[0]
            x_velocity = ((x_position_after - x_position_before)
                          / self.dt)
            ob = self._get_obs()

            reward_ctrl = - 0.1 * np.square(action).sum()
            reward_vel = -np.abs(x_velocity - self.target_velocity)
            reward = reward_ctrl + reward_vel
            #reward_run = (xposafter - xposbefore)/self.dt
            #reward = reward_ctrl + reward_run
            done = False
            self.timestep += 1
            return ob, reward, done, dict()
        elif self.layer == 2:
            target_v = action
            dist_travelled = -1*self.data.qpos[0]
            for _k in range(self.layer2_itr):
                ll_action = self.ll_agent.select_action(self._get_obs(layer=1, t_vel=np.array([target_v[-1]])))
                self.do_simulation(ll_action, self.frame_skip)
            dist_travelled += self.data.qpos[0]
            self.target_delta_x_pos = self.target_delta_x_pos - dist_travelled
            #reward_ctrl = - 0.1 * np.square(action).sum()
            #reward_vel = -np.abs(x_velocity - self.target_velocity)
            if np.abs(self.original_tdxp) < 0.1:
                surr = 0.1
            else:
                surr = self.original_tdxp
            reward_pos = -np.abs(self.target_delta_x_pos/surr)
            reward = reward_pos #reward_vel
            #reward_run = (xposafter - xposbefore)/self.dt
            #reward = reward_ctrl + reward_run
            ob = self._get_obs(layer=2)
            done = False
            self.timestep += 1
            return ob, reward, done, dict()

    def _get_obs(self, layer=None, t_vel=None):
        if layer is None:
            layer = self.layer

        if layer == 1:
            if t_vel is not None:
                return np.concatenate([
                    self.sim.data.qpos.flat[1:],
                    self.sim.data.qvel.flat,
                    t_vel
                ])
            return np.concatenate([
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
                np.array([self.target_velocity])
            ])
        elif layer == 2:
            return np.concatenate([
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
                np.array([self.target_delta_x_pos])
            ])

    def reset_model(self):
        self.timestep = 0
        if self.layer == 1:
            self.target_velocity = np.random.uniform(low=-1.0, high=2.5)
        elif self.layer == 2:
            self.target_delta_x_pos = np.random.uniform(low=-20.0, high=50.0)
            self.original_tdxp = deepcopy(self.target_delta_x_pos)

        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        return obs

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
