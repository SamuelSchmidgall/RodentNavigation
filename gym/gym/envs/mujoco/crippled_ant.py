import numpy as np
from gym import utils
from copy import deepcopy
from gym.envs.mujoco import mujoco_env

class CrippledAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)
        self.cripple = True
        self.crippled_leg_id = None
        self.base_color = deepcopy(self.model.geom_rgba)
        self.cripple_prob = [0.4, 0.15, 0.15, 0.15, 0.15] # None .40, 0-4 0.15 each

    def step(self, action):
        if type(action) == np.ndarray and type(action[0]) == np.ndarray:
            action = action[0]
        limit_penalty = [1 if _ == 1.0 else 0 for _ in np.abs(action)]
        limit_penalty = sum(limit_penalty)*-0.1
        try:
            crippled_leg = self.crippled_leg_id
        except AttributeError:
            self.cripple = False
            self.crippled_leg_id = None
            crippled_leg = self.crippled_leg_id
            self.cripple_prob = [0.4, 0.15, 0.15, 0.15, 0.15]
        if self.cripple and crippled_leg is not None:
            action[self.crippled_leg_id*2:(self.crippled_leg_id+1)*2] *= 0.0

        z_velocity = self.data.body_xvelp[self.model._body_name2id['torso']][0]
        self.do_simulation(action, self.frame_skip)
        #ctrl_cost = .5 * np.square(action).sum()
        #contact_cost = 0.5 * 1e-3 * np.sum(
        #    np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.01

        reward = survive_reward + z_velocity + limit_penalty#forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and 0.2 <= state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            crippled=self.cripple and crippled_leg is not None
            #reward_forward=forward_reward,
            #reward_ctrl=-ctrl_cost,
            #reward_contact=-contact_cost,
            #reward_survive=survive_reward
        )

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        # choose leg to cripple from None to 0-3
        self.model.geom_rgba[:] = self.base_color[:]
        self.crippled_leg_id = np.random.choice([None, 0, 1, 2, 3], p=self.cripple_prob)
        if self.cripple and self.crippled_leg_id is not None:
            self.model.geom_rgba[2+((self.crippled_leg_id+3)%4)*3] = np.array([1, 0.1, 0.1, 1])
            self.model.geom_rgba[2+((self.crippled_leg_id+3)%4)*3 + 1] = np.array([1, 0.1, 0.1, 1])
            self.model.geom_rgba[2+((self.crippled_leg_id+3)%4)*3 + 2] = np.array([1, 0.1, 0.1, 1])

        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5