import math
import numpy as np
from gym import utils
from gym import spaces
from copy import deepcopy
from gym.envs.mujoco import mujoco_env


def quat_to_euler(quaternion):
    euler = list()
    euler.append(math.atan2(2*(quaternion[0]*quaternion[1]+quaternion[2]*quaternion[3]),
                            1-2*(quaternion[1]*quaternion[1]+quaternion[2]*quaternion[2])))
    euler.append(math.asin(2*(quaternion[0]*quaternion[2]-quaternion[3]*quaternion[1])))
    euler.append(math.atan2(2*(quaternion[0]*quaternion[3]+quaternion[1]*quaternion[2]),
                            1-2*(quaternion[2]*quaternion[2]+quaternion[3]*quaternion[3])))
    return euler
"""
void quat_to_euler(float q[4], float e[3]) {
  e[0] = deg(atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])));
  e[1] = deg(asin(2 * (q[0] * q[2] - q[3] * q[1])));
  e[2] = deg(atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3])));
}
"""



class HurdleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        self.task = [
            'walking', 'standing', 'stairs', 'crawl_space', 'none', 'maze']
        self.num_steps = 10
        self.step_height = 0.15
        self.second_num_steps = 9
        self.stair_direction = 'left'

        self.timesteps = 0
        self.max_ts = 30000000
        self.task = 'walking_rotat_mag'

        self.prev_theta = 1.57
        self.target_trunk_vel = np.array([1.0, 0.0, 0.0])
        self.target_trunk_vel_stairs = np.array([1.0, 0.0, 0.0])
        self.target_trunk_vel_standing = np.array([0.0, 0.0, 0.0])

        if self.task == "walking_rotat_mag":

            """ Mag """

            self.smoothing_t = 0
            self.smoothing_delta_t = 200
            self.target_vel_yaw = np.random.uniform(-1, 1)
            self.prev_target_vel_yaw = self.target_vel_yaw

            self.target_vel = np.array(
                [np.random.uniform(0.0, 2.0), np.random.uniform(0.0, 2.0)])
            self.prev_target_vel = self.target_vel

            """ Mag """

        self.generate_xml()
        self.prev_joint1 = np.zeros(19)
        self.prev_joint2 = np.zeros(19)
        self.action_previous = np.zeros(12)

        mujoco_env.MujocoEnv.__init__(self, 'hurdle_gen.xml', 5)
        if self.task == "walking_rotat_mag":
            trunk_orientation = self.data.sensordata[6:10]
            trunk_orientation_rpy = quat_to_euler(trunk_orientation)
            self.prev_target_vel_yaw = trunk_orientation_rpy[2]
            self.prev_target_vel = self.data.body_xvelp[1][:2]
        #self.action_space = spaces.Box(
        #    low=np.ones(1)*-1, high=np.ones(1)*1)
        #self.observation_space = spaces.


    def step(self, action):
        #action_noise = np.random.rand(12)*0.01
        #observation_noise = np.random.rand(19*3)*0.01
        #action += action_noise

        action *= 25

        action = np.clip(action, -25, 25)
        self.do_simulation(action, self.frame_skip)

        self.timesteps += 1

        try:
            _task = self.task
        except AttributeError:
            self.task = 'none'
            _task = self.task
            self.prev_joint1 = np.zeros(19)
            self.prev_joint2 = np.zeros(19)


        trunk_height = self.data.body_xpos[1][2]

        if self.task == "stairs":
            body_id = {'front_right_foot':10, 'front_left_foot':6 ,'back_left_foot': 14, 'back_right_foot': 18}
            non_foot_bodies = [_ for _ in range(18) if _ not in body_id.values()]
            non_foot_contact_forces = \
                np.sum(np.abs(np.array([self.data.cfrc_ext[_][:3] for _ in non_foot_bodies])))/300
            x_vel = np.abs(self.data.body_xvelp[1][2])
            torque_penalty = np.sum(np.abs(action))/(25*12)
            action_difference = np.sum(np.abs(self.action_previous - action))/(25*12)

            target_vel_delta = 0.5*np.sum(np.square(self.target_trunk_vel_stairs - self.data.body_xvelp[1]))\
                    if trunk_height < 0.0 or self.data.body_xvelp[1][0] > self.target_trunk_vel_stairs[0] else 0

            reward = (-1*(target_vel_delta*2 + action_difference*0.2 + torque_penalty*0.2
                          + x_vel*0.3 + non_foot_contact_forces*0.5) + (trunk_height+0.2)*4 + 1)/2

            game_over = trunk_height < -0.35
            if game_over:
                reward = -40

        elif self.task == "walking":
            body_id = {'front_right_foot':10, 'front_left_foot':6 ,'back_left_foot': 14, 'back_right_foot': 18}
            non_foot_bodies = [_ for _ in range(19) if _ not in body_id.values()]
            non_foot_contact_forces = \
                np.sum(np.abs(np.array([self.data.cfrc_ext[_][:3] for _ in non_foot_bodies])))/300
            x_vel = np.abs(self.data.body_xvelp[1][2])
            torque_penalty = np.sum(np.abs(action))/(25*12)
            action_difference = np.sum(np.abs(self.action_previous - action))/(25*12)

            target_vel_delta = 0.5*np.sum(np.square(self.target_trunk_vel - self.data.body_xvelp[1]))

            foot_clearance_cost = 0.05 * sum([(self.data.site_xpos[_ + 1][2] - 0.07) ** 2 for _ in range(4)])
            foot_slip_cost = 0.001*sum([(self.data.site_xvelp[_ + 1][2] - 0.07) ** 2 for _ in range(4)])

            reward = (-1*(target_vel_delta*3 + action_difference*0.2
                    + torque_penalty*0.2 + x_vel*0.3 + non_foot_contact_forces*0.5)
                      + foot_clearance_cost + foot_slip_cost)/2 + 4 + (1+trunk_height)

            game_over = trunk_height < -0.25
            if game_over:
                reward = -40

        elif self.task == "walking_rotat_mag":
            self.smoothing_t += 1
            s_trans = min((self.smoothing_t/self.smoothing_delta_t), 1)
            t_yaw = s_trans*self.target_vel_yaw + (1-s_trans)*self.prev_target_vel_yaw
            t_x = s_trans*self.target_vel[0] + (1-s_trans)*self.prev_target_vel[0]
            t_y = s_trans*self.target_vel[1] + (1-s_trans)*self.prev_target_vel[1]
            trunk_linear_velocity = self.data.sensordata[38:41]

            #trunk_angular_vel = self.data.sensordata[0:3]
            trunk_orientation = self.data.sensordata[6:10]
            trunk_orientation_rpy = quat_to_euler(trunk_orientation)

            body_cost_roll = abs(trunk_orientation_rpy[0] - 0)
            body_cost_pitch = abs(trunk_orientation_rpy[1] - 0)
            body_cost_yaw = abs(trunk_orientation_rpy[2] - t_yaw)

            body_reward_yaw = math.pow(body_cost_yaw, 2)
            body_reward_roll = math.pow(body_cost_roll, 2)
            body_reward_pitch = math.pow(body_cost_pitch, 2)

            velocity_error_x = trunk_linear_velocity[0] - t_x
            velocity_error_y = trunk_linear_velocity[1] - t_y

            velocity_cost = np.sqrt(pow(velocity_error_x, 2)
                + pow(velocity_error_y, 2) + pow(body_cost_yaw, 2))
            cost_control = np.square(action/25).sum() / 12

            reward_run_x = max(0.001, velocity_cost)
            body_reward_yaw = max(0.001, body_reward_yaw)
            body_reward_pitch = max(0.001, body_reward_pitch)
            cost_ctrl = max(0.001, cost_control)
            contact_forces = self.data.sensordata[34:38]
            contact_forces_scaled = np.array([min(x, 50) for x in contact_forces]) / 50.0
            cost_contacts = np.square(contact_forces_scaled).sum() / 4.0
            cost_contact = max(0.001, cost_contacts)

            body_id = {'front_right_foot':10, 'front_left_foot':6 ,'back_left_foot': 14, 'back_right_foot': 18}
            non_foot_bodies = [_ for _ in range(19) if _ not in body_id.values()]
            non_foot_contact_forces = \
                np.sum(np.abs(np.array([self.data.cfrc_ext[_][:3] for _ in non_foot_bodies])))/300
            #x_vel = np.abs(self.data.body_xvelp[1][2])
            torque_penalty = np.sum(np.abs(action))/(25*12)
            action_difference = np.sum(np.abs(self.action_previous - action))/(25*12)
            foot_slip_cost = 0.001*sum([(self.data.body_xpos[body_id[
                list(body_id.keys())[_]]][2] - 0.07) ** 2 for _ in range(4)])
            foot_clearance_cost = 0.05*sum([(self.data.body_xpos[body_id[
                list(body_id.keys())[_]]][2] - 0.07) ** 2 for _ in range(4)])

            ctrl_penalty = -1*0.5*(reward_run_x + body_reward_roll*0.1 + body_reward_yaw*0.1 + body_reward_pitch*0.1)
            energy_penalty = -1 * (action_difference*0.01 + torque_penalty*0.001 + cost_ctrl*0.01 +
                non_foot_contact_forces*0.005 + foot_clearance_cost*0.2 + foot_slip_cost*0.2 + cost_contact*0.1)
            reward = ctrl_penalty + energy_penalty + 5

            game_over = trunk_height < -0.25
            if game_over:
                reward = -40

        else:
            reward = None
            game_over = True


        obs = self._get_obs()
        self.prev_joint2 = deepcopy(self.prev_joint1)
        self.prev_joint1 = deepcopy(self.data.qpos)

        return obs, reward, game_over, {"distance_x": self.data.body_xpos[1][0]}

    def reset_model(self):

        self.smoothing_t = 0

        self.prev_joint1 = np.zeros(19)
        self.prev_joint2 = np.zeros(19)
        self.action_previous = np.zeros(12)

        #self.prev_theta = 1.57
        self.target_vel = np.array(
            [np.random.uniform(0.0, 2.0), np.random.uniform(0.0, 2.0)])

        #self.target_vel_yaw = np.random.uniform(-1, 1)
        #trunk_orientation = self.data.sensordata[6:10]
        #trunk_orientation_rpy = quat_to_euler(trunk_orientation)
        #self.prev_target_vel_yaw = trunk_orientation_rpy[2]

        #self.prev_target_vel = self.data.body_xvelp[1][:2]
        # add stochasticity
        return self._get_obs()

    def generate_xml(self):
        with open('/home/sam/gym/gym/envs/mujoco/assets/hurdle_gen.xml', 'r') as f:
            xml = f.read()

        if self.task == "stairs":

            prim_wall_left =\
            """<body>\n
              <geom name="primary_wall_left" size="{} 0.5 5" pos="{} 3 0" type="box" rgba="0.5 0.5 0.5 0.5" conaffinity="2"/>\n
            </body>\n""".format(10 if self.stair_direction == 'right' else 7,
                                3 if self.stair_direction == 'right' else 0)

            prim_wall_right =\
            """<body>\n
              <geom name="primary_wall_right" size="{} 0.5 5." pos="{} -3 0" type="box" rgba="0.5 0.5 0.5 0.5" conaffinity="2"/>\n
            </body>\n""".format(7 if self.stair_direction == 'right' else 10,
                                0 if self.stair_direction == 'right' else 3)

            prim_wall_behind =\
            """<body>\n
              <geom name="primary_wall_behind" size="0.5 3.5 5" pos="-6.5 0 0" type="box" rgba="0.5 0.5 0.5 0.5" conaffinity="2"/>\n
            </body>\n"""

            stair_wall_left =\
            """<body>\n
              <geom name="stair_wall_left" size="0.5 7.5 8" pos="12.5 {} 0" type="box" rgba="0.5 0.5 0.5 0.5" conaffinity="2"/>\n
            </body>\n""".format(-4 if self.stair_direction == 'right' else 4)

            stair_wall_behind =\
            """<body>\n
              <geom name="stair_wall_behind" size="3.5 0.5 8" pos="9.5 {} 0" type="box" rgba="0.5 0.5 0.5 0.5" conaffinity="2"/>\n
            </body>\n""".format(-12 if self.stair_direction == 'right' else 12)

            stair_wall_right =\
            """<body>\n
              <geom name="stair_wall_right" size="0.5 5 8" pos="6.5 {} 0" type="box" rgba="0.5 0.5 0.5 0.5" conaffinity="2"/>\n
            </body>\n""".format(-7.5 if self.stair_direction == 'right' else 7.5)

            stair_platform =\
            """<body>\n
              <geom name="second_stair_platform" size="2.5 2.5 2.001" pos="9.5 0 0" type="box" conaffinity="2"/>\n
            </body>\n"""

            direction = -1 if self.stair_direction == 'right' else 1

            final_stair_platform =\
            """<body>\n
              <geom name="final_stair_platform" size="2.5 2.5 {}" pos="9.5 {} 0" type="box" conaffinity="2"/>\n
            </body>\n""".format(self.step_height*self.num_steps - 0.5 +
                                self.step_height*(self.second_num_steps-1) + 0.01, direction*9)

            step_str = prim_wall_left + prim_wall_right + prim_wall_behind + stair_wall_left +\
                    stair_wall_behind + stair_wall_right + stair_platform + final_stair_platform

            direction *= -1

            for step in range(self.second_num_steps):
                step_block =\
                """
                <body>\n
                    <geom name="second_stair_block{}" size="2.5 0.5 {}" pos="9.5 {} 0" type="box" conaffinity="2"/>\n
                </body>\n""".format(step, 2.0+0.25*step, (-3+-1*0.5*step)*direction)
                step_str += step_block

            for step in range(self.num_steps):
                step_str += "    <body>\n"
                step_str += "        <geom name=\"generated_stair_block{}\" size=\"{}\" pos=\"{}\" type=\"box\" condim=\"3\" friction=\"0.5 .1 .1\" conaffinity=\"2\"/>\n"\
                    .format(step, "0.5 2.5 {}".format((step+1)*self.step_height), "{} 0 -0.6".format(2.0+0.5*step)
                )
                step_str += "    </body>\n"
            xml = xml.replace("<!-- REPLACE ME -->", step_str)

        elif self.task == "none":
            pass

        elif self.task == 'maze':
            maze ="""<geom conaffinity="1" contype="1" material="" name="block_0_0" pos="-3.0 -3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_0_1" pos="0.0 -3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_0_2" pos="3.0 -3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_0_3" pos="6.0 -3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_0_4" pos="9.0 -3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_0_5" pos="12.0 -3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_0_6" pos="15.0 -3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_0_7" pos="18.0 -3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_0_8" pos="21.0 -3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_1_0" pos="-3.0 0.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_1_6" pos="15.0 0.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_1_8" pos="21.0 0.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_2_0" pos="-3.0 3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_2_1" pos="0.0 3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_2_2" pos="3.0 3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_2_3" pos="6.0 3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_2_4" pos="9.0 3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_2_6" pos="15.0 3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_2_8" pos="21.0 3.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_3_0" pos="-3.0 6.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_3_4" pos="9.0 6.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_3_6" pos="15.0 6.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_3_8" pos="21.0 6.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_4_0" pos="-3.0 9.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_4_2" pos="3.0 9.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_4_4" pos="9.0 9.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_4_6" pos="15.0 9.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_4_8" pos="21.0 9.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_5_0" pos="-3.0 12.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_5_2" pos="3.0 12.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_5_6" pos="15.0 12.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_5_8" pos="21.0 12.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_6_0" pos="-3.0 15.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_6_2" pos="3.0 15.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_6_3" pos="6.0 15.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_6_4" pos="9.0 15.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_6_5" pos="12.0 15.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_6_6" pos="15.0 15.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_6_8" pos="21.0 15.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_7_0" pos="-3.0 18.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_7_8" pos="21.0 18.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_8_0" pos="-3.0 21.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_8_1" pos="0.0 21.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_8_2" pos="3.0 21.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_8_3" pos="6.0 21.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_8_4" pos="9.0 21.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_8_5" pos="12.0 21.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_8_6" pos="15.0 21.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_8_7" pos="18.0 21.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/><geom conaffinity="1" contype="1" material="" name="block_8_8" pos="21.0 21.0 1.5" rgba="0.59 0.31 0.02 1" size="1.5 1.5 1.5" type="box"/>"""

            xml = xml.replace("<!-- REPLACE ME -->", maze)


        with open('/home/sam/gym/gym/envs/mujoco/assets/hurdle_gen.xml', 'w') as f:
            f.write(xml)


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def _get_obs(self):
        joint_angles = self.data.qpos
        trunk_height = self.data.body_xpos[1][2]

        #todo: what is sensor data?
        sensor_data = self.data.sensordata
        if self.task == "walking_rotat_mag":
            #self.target_vel_mag = np.ones(1)
            #self.target_rotational_vel = np.array([np.random.uniform(-0.3, 0.3)])

            s_trans = min((self.smoothing_t / self.smoothing_delta_t), 1)
            t_yaw = s_trans * self.target_vel_yaw + (1 - s_trans) * self.prev_target_vel_yaw
            t_x = s_trans * self.target_vel[0] + (1 - s_trans) * self.prev_target_vel[0]
            t_y = s_trans * self.target_vel[1] + (1 - s_trans) * self.prev_target_vel[1]

            sensor_data = np.concatenate((self.target_vel, sensor_data))
            s_data = np.concatenate((self.prev_joint1, self.prev_joint2, joint_angles, sensor_data))
            s_data = np.concatenate((np.array([trunk_height, t_yaw, t_x, t_y]), s_data))
            return s_data

        if self.task == "walking":
            sensor_data = np.concatenate((self.target_trunk_vel, sensor_data))

        s_data = np.concatenate((self.prev_joint1, self.prev_joint2, joint_angles, sensor_data))
        s_data = np.concatenate((np.array([trunk_height]), s_data))
        return s_data



# axis = [0, 0, 1]
# sin_half_angle = np.sqrt(
#    axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
# angle = 2 * np.arctan2(sin_half_angle, self.data.body_xquat[1][0])
# angle_change_penalty = np.abs(angle-self.prev_theta)
# self.prev_theta = angle

# target_rot_vel_delta = 0.5 * np.sum(np.square(
#    angle - self.target_angle))*(self.timesteps/self.max_ts)

# reward = (-1*((target_vel_delta*6)/2 + action_difference*0.2
#        + torque_penalty*0.2 + x_vel*0.3 + non_foot_contact_forces*0.5)
#        + foot_clearance_cost + foot_slip_cost)/2 + 4 + (1+trunk_height)



