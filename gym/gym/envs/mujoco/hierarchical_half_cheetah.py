import torch
import pickle
import numpy as np
from gym import utils
from gym import spaces
from copy import deepcopy
from gym.envs.mujoco import mujoco_env


class HierarchicalHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, policy=None):
        self.controller_type = "red_box"
        if self.controller_type == "velocity":
            self.target_vel_range = (-1, 2.5)
            self.target_velocity = np.random.uniform(
                self.target_vel_range[0], self.target_vel_range[1])

        elif self.controller_type == "position":
            self.vel_controller_time = 5
            self.target_pos_range = (-20, 80)
            self.target_position = np.random.uniform(
                self.target_pos_range[0], self.target_pos_range[1])
            if policy is None:
                self.low_level_controller_loc = \
                    "/home/sam/PycharmProjects/RodentNavigation/Networks/PolicyTypes/model_layer2_dump.pkl"
                with open(self.low_level_controller_loc, "rb") as f:
                    self.ll_controller = pickle.load(f)
            else:
                self.ll_controller = policy
            self.original_tdxp = deepcopy(self.target_position)

        elif self.controller_type == "red_box":
            self.target_vel_range = (-1, 2.5)
            self.target_velocity = np.random.uniform(
                self.target_vel_range[0], self.target_vel_range[1])
            self.vel_controller_time = 5
            self.pos_controller_time = 5
            self.red_box_range = (150, 300)
            self.red_box_pos = np.random.uniform(
                self.red_box_range[0], self.red_box_range[1])
            self.original_rbp = deepcopy(self.red_box_pos)
            self.low_level_controller_loc = \
                "/home/sam/PycharmProjects/RodentNavigation/Networks/PolicyTypes/HalfCheetah/models/model_layer2_dump.pkl"
            with open(self.low_level_controller_loc, "rb") as f:
                self.ll_controller = pickle.load(f)
            self.low_level_controller_loc2 = \
                "/home/sam/PycharmProjects/RodentNavigation/Networks/PolicyTypes/HalfCheetah/models/model_layer3_dump.pkl"
            with open(self.low_level_controller_loc2, "rb") as f:
                self.ll2_controller = pickle.load(f)
            self.generate_red_target_box()
            mujoco_env.MujocoEnv.__init__(self, 'hierarchical_half_cheetah_cpy.xml', 1)

        if self.controller_type != "red_box":
            mujoco_env.MujocoEnv.__init__(self, 'hierarchical_half_cheetah.xml', 1)

        utils.EzPickle.__init__(self)
        if self.controller_type == "position":
            self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([2.5]), dtype=np.float32)

        elif self.controller_type == "red_box":
            self.action_space = spaces.Box(low=np.array([-10.0]), high=np.array([70.0]), dtype=np.float32)

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
            dist_travelled = -1*self.data.qpos[0]
            for _i in range(self.vel_controller_time):
                vel_command = action
                obs = self._get_obs()
                obs[-1] = vel_command[0]
                obs = torch.FloatTensor(obs).reshape((1, obs.shape[0]))
                act = self.ll_controller.select_action(obs)[0]
                self.do_simulation(act, self.frame_skip)
            dist_travelled += self.data.qpos[0]
            #prev_target_pos = self.target_position
            self.target_position -= dist_travelled
            reward_pos = -np.abs(np.abs(self.target_position)
                /np.abs(self.original_tdxp if abs(self.original_tdxp) > 1 else 1))
            distance_reward = reward_pos
            done = False
            obs = self._get_obs()
            return obs, distance_reward, done, dict()
        elif self.controller_type == "red_box":
            pos_control = action
            for _j in range(self.pos_controller_time):
                obs_pos = self._get_obs(ctype="velocity")
                obs_pos[-1] = pos_control[0]
                obs_pos = torch.FloatTensor(obs_pos).reshape((1, obs_pos.shape[0]))
                vel_command = self.ll2_controller.select_action(obs_pos)[0]
                for _i in range(self.vel_controller_time):
                    obs = self._get_obs()
                    obs[-1] = vel_command[0]
                    obs = torch.FloatTensor(obs).reshape((1, obs.shape[0]))
                    act = self.ll_controller.select_action(obs)[0]
                    self.do_simulation(act, self.frame_skip)
            after_red_box_pos = self.red_box_pos
            reward = -np.abs(np.abs(after_red_box_pos)
                    / np.abs(self.original_rbp if abs(self.original_rbp) > 1 else 1))
            game_over = False
            obs = self._get_obs()
            return obs, reward, game_over, dict()

        else:
            raise TypeError("No such controller type: {}".format(self.controller_type))

    def _get_obs(self, ctype=None):
        if (ctype is None and self.controller_type == "velocity") or ctype == "velocity":
            return np.concatenate([
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
                np.array([self.target_velocity])
            ])

        elif self.controller_type == "position":
            return np.concatenate([
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
                np.array([self.target_position - self.data.qpos[0]])
            ])

        elif self.controller_type == "red_box":
            # return SOME IMAGE
            fp_image = self.sim.render(
                width=64, height=64, camera_name='track1', depth=False)
            return fp_image

    def reset_model(self):
        if self.controller_type == "velocity":
            self.target_velocity = np.random.uniform(
                self.target_vel_range[0], self.target_vel_range[1])

        elif self.controller_type == "position":
            self.target_position = np.random.uniform(
                self.target_pos_range[0], self.target_pos_range[1])
            self.original_tdxp = deepcopy(self.target_position)

        elif self.controller_type == "red_box":
            self.red_box_pos = np.random.uniform(
                self.red_box_range[0], self.red_box_range[1])
            self.original_rbp = deepcopy(self.red_box_pos)
            self.generate_red_target_box()
            mujoco_env.MujocoEnv.__init__(self, 'hierarchical_half_cheetah_cpy.xml', 1)
            utils.EzPickle.__init__(self)
            self.action_space = spaces.Box(low=np.array([-10.0]), high=np.array([70.0]), dtype=np.float32)

        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def generate_red_target_box(self):
        with open("/home/sam/Downloads/gym/gym/envs/mujoco/assets/hierarchical_half_cheetah.xml", "r") as f:
            xml_content = f.read()

        red_box = \
        """    <body name="target_box" pos="0 0 0">
            <geom name="r_box" pos="195.6839975551898 0 0" size="1 1 0.5" contype="1" type="box"/>
            </body>""".format(self.original_rbp+1)

        xml_content = xml_content.replace("<!-- Insert here -->", red_box)
        with open("/home/sam/Downloads/gym/gym/envs/mujoco/assets/hierarchical_half_cheetah_cpy.xml", "w") as f:
            f.write(xml_content)


    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
