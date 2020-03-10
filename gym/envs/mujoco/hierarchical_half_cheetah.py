import pickle
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HierarchicalHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.controller_type = "velocity"
        if self.controller_type == "velocity":
            self.target_vel_range = (-1, 2.5)
            self.target_velocity = np.random.uniform(
                self.target_vel_range[0], self.target_vel_range[1])
        elif self.controller_type == "position":
            self.vel_controller_time = 5
            self.target_pos_range = (-20, 80)
            self.target_position = np.random.uniform(
                self.target_pos_range[0], self.target_pos_range[1])
            self.low_level_controller_loc = "low_level.pkl"
            with open(self.low_level_controller_loc, "rb") as f:
                self.ll_controller = pickle.load(f)
        elif self.controller_type == "red_box":
            self.red_box_range = (50, 300)
            self.red_box_pos = np.random.uniform(
                self.red_box_range[0], self.red_box_range[1])

        mujoco_env.MujocoEnv.__init__(self, 'hierarchical_half_cheetah.xml', 1)
        utils.EzPickle.__init__(self)

    def step(self, action,):
        if self.controller_type == "velocity":
            xposbefore = self.sim.data.qpos[0]
            self.do_simulation(action, self.frame_skip)
            xposafter = self.sim.data.qpos[0]
            ob = self._get_obs()
            reward_ctrl = -0.1*np.square(action).sum()
            reward_run = (xposafter - xposbefore)/self.dt
            reward = reward_ctrl + -abs(reward_run - self.target_velocity) + 1
            done = False
            return ob, reward, done, dict()
        elif self.controller_type == "position":
            dist_before = self.data.body_xpos[1][0]
            for _i in range(self.vel_controller_time):
                vel_command = action
                obs = self._get_obs()
                obs[-1] = vel_command[0]
                action = self.ll_controller(obs)
                self.do_simulation(action, self.frame_skip)
            dist_after = self.data.body_xpos[1][0]
            goal_progress = (dist_after - dist_before)
            prev_target_pos = self.target_position
            self.target_position -= goal_progress
            distance_reward = goal_progress*np.sign(
                np.abs(prev_target_pos) - np.abs(self.target_position))
            done = False
            obs = self._get_obs()
            return obs, distance_reward, done, dict()
        else:
            raise TypeError("No such controller type: {}".format(self.controller_type))

    def _get_obs(self):
        if self.controller_type == "velocity":
            return np.concatenate([
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
                np.array([self.target_velocity])
            ])

        elif self.controller_type == "position":
            return np.concatenate([
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
                np.array([self.target_position - self.data.body_xpos[1][0]])
            ])

        elif self.controller_type == "red_box":
            # return SOME IMAGE
            return np.concatenate([
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
                np.array([self.target_position - self.data.body_xpos[1][0]])
            ])

    def reset_model(self):
        if self.controller_type == "velocity":
            self.target_velocity = np.random.uniform(
                self.target_vel_range[0], self.target_vel_range[1])

        elif self.controller_type == "position":
            self.target_position = np.random.uniform(
                self.target_pos_range[0], self.target_pos_range[1])

        elif self.controller_type == "red_box":
            self.red_box_pos = np.random.uniform(
                self.red_box_range[0], self.red_box_range[1])

        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def generate_red_target_box(self):
        with open("hierarchical_half_cheetah.xml", "r") as f:
            xml_content = f.read()

        red_box = \
        """    <body name="target_box" pos="{} 0 0">
              
            
        """
        with open("hierarchical_half_cheetah.xml", "r") as f:
            xml_content = f.read()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
