import math
import torch
import pickle
import numpy as np
from torch import nn
from scipy import signal
import torch.optim as optim
from torch.distributions import Normal



def _uniform_init(tensor, param=3e-3):
    return tensor.data.uniform_(-param, param)

def layer_init(layer, weight_init = _uniform_init, bias_init = _uniform_init ):
    weight_init(layer.weight)
    bias_init(layer.bias)

def uniform_init(layer):
    layer_init(layer, weight_init = _uniform_init, bias_init = _uniform_init )


class SpinalActorCritic(nn.Module):
    def __init__(self, policy, net_type="linear",
            policy_learning_rate=0.0003, value_learning_rate=0.0003, optimizer=optim.Adam):

        super(SpinalActorCritic, self).__init__()
        self.policy = policy
        self.net_type = net_type

        # generates value function with same topology
        self.value_function = policy.generate_value_function()

        policy_params, value_params = self.params()

        self.value_optim = optimizer(params=value_params, lr=value_learning_rate)
        self.policy_optim = optimizer(params=policy_params, lr=policy_learning_rate)

    def forward(self, x):
        return self.policy(x)

    def value(self, x):
        return self.value_function(x)

    def params(self):
        return self.policy.params(), self.value_function.params()

    def optimize(self, policy_loss, value_loss):
        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        if self.policy.plastic:
            nn.utils.clip_grad_norm(self.policy.parameters(), 10)
        self.policy_optim.step()

    def reset(self):
        self.policy.reset()




class ReplayMemory:
    def __init__(self,):
        self.actions_top = list()
        self.actions_bottom = list()

        self.rewards_top = list()
        self.rewards_bottom = list()

        self.log_probs_top = list()
        self.log_probs_bottom = list()

        self.reset_flags_top = list()
        self.reset_flags_bottom = list()

        self.sensor_states_top = list()
        self.sensor_states_bottom = list()

    def clear(self):
        self.actions_top = list()
        self.actions_bottom = list()

        self.rewards_top = list()
        self.rewards_bottom = list()

        self.log_probs_top = list()
        self.log_probs_bottom = list()

        self.reset_flags_top = list()
        self.reset_flags_bottom = list()

        self.sensor_states_top = list()
        self.sensor_states_bottom = list()


class SpinalPlasticityModule(nn.Module):
    def __init__(self, dim_1, dim_2, plasticity=True, plasticity_type="neuromodulated_linear"):
        super(SpinalPlasticityModule, self).__init__()
        self.plasticity = plasticity
        self.plasticity_type = plasticity_type
        self.hebbian_trace = torch.zeros(size=(dim_1, dim_2))
        # create parameters for respective plasticity type
        if self.plasticity_type == "neuromodulated_linear":
            # linear feedforward weights
            self.forward_weights = nn.Linear(dim_1, dim_2)
            # modulatory update rule weights
            self.eta_fan_in = nn.Linear(dim_2, self.COMPRESS_DIM)
            self.eta_fan_out = nn.Linear(self.COMPRESS_DIM, dim_2)
            # alpha plasticity modulation weights
            self.alpha_plasticity = torch.rand(size=(dim_1, dim_2))*0.01
            # uniform initialize weights
            uniform_init(self.eta_fan_in)
            uniform_init(self.eta_fan_out)
            uniform_init(self.forward_weights)

    def forward(self, x):
        pass

    def update_trace(self):
        pass


class SpinalNetworkModule(nn.Module):
    def __init__(self, dim_1, dim_2, module_type='continuous_reinforcement', optional_args=None):
        super(SpinalNetworkModule, self).__init__()

        self.dimension_1 = dim_1
        self.dimension_2 = dim_2
        self.module_type = module_type

        if module_type not in self.module_types:
            raise Exception("{} is not a module type".format(module_type))

        if module_type == "continuous_reinforcement":
            # linear feedforward
            self.activation = torch.tanh
            self.linear = nn.Linear(dim_1, dim_2)
            nn.init.xavier_normal_(self.linear.weight)
            self.linear.bias.data *= 0

        elif module_type == "continuous_reinforcement_vision":
            # convolutional reinforcement vision feedforward
            stride = 2
            kernel = 5
            self.pooling = False
            self.activation = torch.relu
            if 'kernel' in optional_args:
                kernel = optional_args['kernel']
            if 'stride' in optional_args:
                stride = optional_args['stride']
            if 'activation' in optional_args:
                self.activation = optional_args['activation']
            if 'pooling' in optional_args: # bool
                self.pooling = optional_args['pooling']
                # if pooling then specify pooling kernel
                self.pool = nn.MaxPool2d(kernel_size=2)
                if 'pooling_kernel' in optional_args:
                    self.pool = nn.MaxPool2d(kernel_size=optional_args['pooling_kernel'])
            # the dimensionality of the hypothetical flattened output
            self.flatten_dimensionlity = -1
            self.convolutional = nn.Conv2d(in_channels=dim_1, out_channels=dim_2, kernel_size=kernel, stride=stride)

        elif module_type == "continuous_reinforcement_vision_final":
            # final mapping for continuous reinforcement vision -- linear embedding
            self.linear = nn.Linear(dim_1, dim_2)
            nn.init.xavier_normal_(self.linear.weight)
            #uniform_init(self.linear)

        elif module_type == "value_module":
            # feedforward for linear value module
            self.activation = torch.tanh
            self.linear = nn.Linear(dim_1, dim_2)
            uniform_init(self.linear)

        elif module_type == "value_module_final":
            # final output for value module
            self.linear = nn.Linear(dim_1, 1)
            uniform_init(self.linear)

        elif module_type == "value_module_vision":
            # value module for vision
            stride = 2
            kernel = 5
            self.pooling = False
            self.activation = torch.relu
            if 'kernel' in optional_args:
                kernel = optional_args['kernel']
            if 'stride' in optional_args:
                stride = optional_args['stride']
            if 'activation' in optional_args:
                self.activation = optional_args['activation']
            if 'pooling' in optional_args['pooling']: # bool
                self.pooling = optional_args['pooling']
                # if pooling then specify pooling kernel
                self.pool = nn.MaxPool2d(kernel_size=2)
                if 'pooling_kernel' in optional_args:
                    self.pool = nn.MaxPool2d(kernel_size=optional_args['pooling_kernel'])
            # the dimensionality of the hypothetical flattened output
            self.flatten_dimensionlity = -1
            self.convolutional = nn.Conv2d(in_channels=dim_1, out_channels=dim_2, kernel_size=kernel, stride=stride)

        elif module_type == "continuous_reinforcement_final":
            # final module for continuous action space reinforcement agent
            self.linear = nn.Linear(dim_1, dim_2)
            nn.init.xavier_normal_(self.linear.weight)
            #nn.init.xavier_normal_(self.linear.bias, 0)
            self.linear.bias.data *= 0

            self.log_std_lin = nn.Parameter(torch.ones(1, dim_2)*-0.5)
            self.log_std_min, self.log_std_max = -20, 2

    @property
    def module_types(self):
        return ['continuous_reinforcement', 'continuous_reinforcement_final', 'continuous_reinforcement_vision',
                'value_module', 'value_module_vision', 'value_module_final', 'continuous_reinforcement_vision_final']

    def forward(self, x):
        if self.module_type in ['value_module', 'continuous_reinforcement']:
            return self.activation(self.linear(x))

        elif self.module_type in ['value_module_final', 'continuous_reinforcement_vision_final']:
            return self.linear(x)

        elif self.module_type in ['continuous_reinforcement_vision', 'value_module_vision']:
            conv = self.convolutional(x)
            if self.pooling:
                conv = self.pool(conv)
            return self.activation(conv)

        elif self.module_type == "continuous_reinforcement_final":
            mean = self.linear(x)
            return mean, torch.clamp(self.log_std_lin.expand_as(mean), min=self.log_std_min, max=self.log_std_max)


class SpinalNeuralNetwork(nn.Module):
    def __init__(self, latent_shape, environment=None, act_space=None, obs_space=None,
                 plasticity=None, network_type="cont_pg_rl", module_arguments=None):
        super(SpinalNeuralNetwork, self).__init__()

        self.environment = environment
        self.network_type = network_type
        if environment is not None:
            ob_space = environment.observation_space.shape[0]
        else:
            ob_space = obs_space

        if environment is not None:
            ac_space = environment.action_space.shape[0]
        else:
            ac_space = act_space

        self.plastic = plasticity
        self.trace_template = \
            torch.zeros(latent_shape[-1], ac_space, requires_grad=False)
        if plasticity is not None:
            if plasticity == "neuromodulated":
                self.trace = torch.zeros(latent_shape[-1], ac_space, requires_grad=False)

                self.fan_in = nn.Linear(ac_space, 1)
                self.fan_out = nn.Linear(1, ac_space)
                self.alpha_plasticity = nn.Parameter(torch.randn(latent_shape[-1], ac_space)*0.01)

        # generate network shape from latent if cont_pg_rl
        self.network_shape = latent_shape
        if network_type == "cont_pg_rl":
            self.network_shape = [ob_space] + latent_shape + [ac_space]

        # if no model arguments provided generate corresponding argument list
        if module_arguments is None:
            module_arguments = [dict() for _ in range(len(self.network_shape))]

        # check validity of network type
        if network_type not in self.network_types:
            raise Exception("{} is not a network type".format(network_type))

        # generate corresponding network modules
        if network_type == "cont_pg_rl":
            """ Continuous policy gradient reinforcement learning """
            self.network_modules = [
                SpinalNetworkModule(dim_1=self.network_shape[_],
                    dim_2=self.network_shape[_ + 1], module_type="continuous_reinforcement",
                    optional_args=module_arguments[_]) for _ in range(len(self.network_shape) - 2)]
            self.network_modules.append(SpinalNetworkModule(dim_1=self.network_shape[-2], dim_2=self.network_shape[-1],
                    module_type="continuous_reinforcement_final", optional_args=module_arguments[-1]))

        elif network_type == "vanilla":
            """ Continuous policy gradient reinforcement learning """
            self.network_modules = [
                SpinalNetworkModule(dim_1=self.network_shape[_],
                    dim_2=self.network_shape[_ + 1], module_type="continuous_reinforcement",
                    optional_args=module_arguments[_]) for _ in range(len(self.network_shape) - 1)]
            self.network_modules[-1].activation = nn.Identity()

        elif network_type == "vision":
            """ Continuous policy gradient reinforcement learning """
            self.network_modules = [
                SpinalNetworkModule(dim_1=self.network_shape[_],
                    dim_2=self.network_shape[_ + 1], module_type="continuous_reinforcement_vision",
                    optional_args=module_arguments[_]) for _ in range(len(self.network_shape) - 2)]
            self.network_modules.append(SpinalNetworkModule(dim_1=self.network_shape[-2], dim_2=self.network_shape[-1],
                    module_type="continuous_reinforcement_vision_final", optional_args=module_arguments[-1]))

    def params(self):
        # generate parameter list
        param_list = list()
        for _ in range(len(self.network_modules)):
            param_list += list(self.network_modules[_].parameters())
        return param_list

    def generate_value_function(self, optional_args=None):
        # generate value function with same topology as policy
        _itr = 0
        value_modules = list()
        if optional_args is None:
            optional_args = [None for _ in range(len(self.network_modules))]

        for _module in self.network_modules:
            value_modules.append(SpinalNetworkModule(dim_1=_module.dimension_1, dim_2=_module.dimension_2,
                module_type=_module.module_type.replace("continuous_reinforcement", "value_module"),
                optional_args=optional_args[_itr]))
            _itr += 1
        d_copy = deepcopy(self)
        d_copy.network_modules = value_modules
        d_copy.plastic = False
        return d_copy

    @property
    def network_types(self):
        return ["cont_pg_rl", "vanilla", "vision"]

    def forward(self, x):
        # feedforward through modules
        pre_syn = torch.clone(x)
        for _module in self.network_modules:
            pre_syn = torch.clone(x)
            x = _module(x)
        if self.plastic:
            x = x[0]
            plastic_component = torch.mm(pre_syn, self.trace*self.alpha_plasticity)
            new_mean = x + plastic_component
            x = new_mean, torch.clamp(self.network_modules[-1].log_std_lin.expand_as(new_mean),
                min=self.network_modules[-1].log_std_min, max=self.network_modules[-1].log_std_max)
            eta = self.fan_out(self.fan_in(x[0]))
            self.trace = torch.clamp(self.trace + eta*torch.mm(pre_syn.t(), x[0]), min=-1, max=1)
        return x

    def reset(self):
        self.trace = torch.clone(self.trace_template)


class SpinalHierarchicalNetwork(nn.Module):
    def __init__(self, top_layer_ac, bottom_layer_ac, top_layer_freq,
        bottom_layer_freq, epochs=10, minibatch_size=1000, timestep_size=4000, entropy_coefficient=0.01):
        super(SpinalHierarchicalNetwork, self).__init__()

        self.lamb = 0.95
        self.gamma = 0.99
        self.ppo_clip = 0.2

        self.epochs = epochs
        self.timestep_size = timestep_size
        self.minibatch_size = minibatch_size
        self.entropy_coefficient = entropy_coefficient

        self.top_layer_freq = top_layer_freq
        self.bottom_layer_freq = bottom_layer_freq

        self.top_layer_actor_critic = top_layer_ac
        self.bottom_layer_actor_critic = bottom_layer_ac

    def forward(self, x, memory, evaluate=False, append=True, top=False, bottom=False, ):
        if not top and not bottom:
            raise RuntimeError("Must be top or bottom")

        if top:
            action_mean, action_log_std = self.top_layer_actor_critic(x)
            action_std = torch.exp(action_log_std)

            distribution = Normal(loc=action_mean, scale=action_std)
            action = distribution.sample()
            log_probabilities = distribution.log_prob(action)
            log_probabilities = torch.sum(log_probabilities, dim=1)
            if append:
                memory.log_probs_top.append(log_probabilities.detach())
            if evaluate:
                return action_mean.detach(), None
            return action, memory
        elif bottom:
            action_mean, action_log_std = self.bottom_layer_actor_critic(x)
            action_std = torch.exp(action_log_std)

            distribution = Normal(loc=action_mean, scale=action_std)
            action = distribution.sample()
            log_probabilities = distribution.log_prob(action)
            log_probabilities = torch.sum(log_probabilities, dim=1)
            if append:
                memory.log_probs_bottom.append(log_probabilities.detach())
            if evaluate:
                return action_mean.detach(), None
            return action, memory
        raise Exception("Forward not top or bottom")


    def evaluate(self, x, old_action, top=False, bottom=False):
        if not top and not bottom:
            raise RuntimeError("Must be top or bottom")

        if top:
            action_mean, action_log_std = self.top_layer_actor_critic(x)
            action_std = torch.exp(action_log_std)

            distribution = Normal(loc=action_mean, scale=action_std)
            log_probabilities = distribution.log_prob(old_action.squeeze(dim=1))
            log_probabilities = torch.sum(log_probabilities, dim=1)

            entropy = distribution.entropy()

            return log_probabilities, entropy
        elif bottom:
            action_mean, action_log_std = self.bottom_layer_actor_critic(x)
            action_std = torch.exp(action_log_std)

            distribution = Normal(loc=action_mean, scale=action_std)
            log_probabilities = distribution.log_prob(old_action.squeeze(dim=1))
            log_probabilities = torch.sum(log_probabilities, dim=1)

            entropy = distribution.entropy()

            return log_probabilities, entropy

    def generalized_advantage_estimation(self, r, v, mask):
        batchsz = v.size(0)

        # v_target is worked out by Bellman equation.
        delta = torch.Tensor(batchsz)
        v_target = torch.Tensor(batchsz)
        adv_state = torch.Tensor(batchsz)

        prev_v = 0
        prev_v_target = 0
        prev_adv_state = 0
        for t in reversed(range(batchsz)):
            # mask here indicates a end of trajectory
            # this value will be treated as the target value of value network.
            # mask = 0 means the immediate reward is the real V(s) since it's end of trajectory.
            # formula: V(s_t) = r_t + gamma * V(s_t+1)
            v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]

            # please refer to : https://arxiv.org/abs/1506.02438
            # for generalized adavantage estimation
            # formula: delta(s_t) = r_t + gamma * V(s_t+1) - V(s_t)
            delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]

            # formula: A(s, a) = delta(s_t) + gamma * lamda * A(s_t+1, a_t+1)
            # here use symbol tau as lambda, but original paper uses symbol lambda.
            adv_state[t] = delta[t] + self.gamma * self.lamb * prev_adv_state * mask[t]

            # update previous
            prev_v_target = v_target[t]
            prev_v = v[t]
            prev_adv_state = adv_state[t]

        # normalize adv_state
        adv_state = (adv_state - adv_state.mean()) / (adv_state.std() + 1e-6)

        return adv_state, v_target

    def learn(self, memory):
        replay_len = len(memory.rewards_bottom)
        minibatch_count = self.timestep_size / self.minibatch_size

        values = self.bottom_layer_actor_critic.value(torch.FloatTensor(memory.sensor_states_bottom)).detach()

        advantages, value_target = self.generalized_advantage_estimation(
            torch.FloatTensor(memory.rewards_bottom).unsqueeze(1), values, torch.FloatTensor(memory.reset_flags_bottom).unsqueeze(1))

        advantages = advantages.detach().numpy()
        value_target = value_target.detach().numpy()

        for _ in range(self.epochs):
            minibatch_indices = list(range(replay_len))
            np.random.shuffle(minibatch_indices)
            minibatches = [minibatch_indices[int(_ * (replay_len/minibatch_count)):
                int((_ + 1) * (replay_len/minibatch_count))] for _ in range(int(minibatch_count))]

            for batch in minibatches:

                mb_states = torch.FloatTensor(np.array(memory.sensor_states_bottom)[batch])
                mb_actions = torch.stack(memory.actions_bottom).index_select(0, torch.LongTensor(batch))
                mb_old_log_probabilities = torch.stack(memory.log_probs_bottom).index_select(0, torch.LongTensor(batch))

                predicted_values = self.bottom_layer_actor_critic.value(mb_states)

                log_probabilities, entropy = self.evaluate(mb_states, mb_actions, bottom=True)

                mb_advantages = torch.FloatTensor(advantages[batch])

                ratio = (log_probabilities - mb_old_log_probabilities.squeeze()).exp()
                min_adv = torch.where(mb_advantages > 0,
                    (1 + self.ppo_clip) * mb_advantages, (1 - self.ppo_clip) * mb_advantages)
                policy_loss = -(torch.min(ratio * mb_advantages, min_adv)).mean() - self.entropy_coefficient*entropy.mean()

                value_loss = (torch.FloatTensor(value_target[batch]) - predicted_values.squeeze()).pow(2).mean()
                self.bottom_layer_actor_critic.optimize(policy_loss, value_loss)

        adv = list()
        adv += [value_target]

        replay_len = len(memory.rewards_top)
        minibatch_count = self.timestep_size / self.minibatch_size

        values = self.top_layer_actor_critic.value(torch.FloatTensor(memory.sensor_states_top)).detach()

        advantages, value_target = self.generalized_advantage_estimation(
            torch.FloatTensor(memory.rewards_top).unsqueeze(1), values, torch.FloatTensor(memory.reset_flags_top).unsqueeze(1))

        advantages = advantages.detach().numpy()
        value_target = value_target.detach().numpy()

        for _ in range(self.epochs):
            minibatch_indices = list(range(replay_len))
            np.random.shuffle(minibatch_indices)
            minibatches = [minibatch_indices[int(_ * (replay_len/minibatch_count)):
                int((_ + 1) * (replay_len/minibatch_count))] for _ in range(int(minibatch_count))]

            for batch in minibatches:

                mb_states = torch.FloatTensor(np.array(memory.sensor_states_top)[batch])
                mb_actions = torch.stack(memory.actions_top).index_select(0, torch.LongTensor(batch))
                mb_old_log_probabilities = torch.stack(memory.log_probs_top).index_select(0, torch.LongTensor(batch))

                predicted_values = self.top_layer_actor_critic.value(mb_states)

                log_probabilities, entropy = self.evaluate(mb_states, mb_actions, top=True)

                mb_advantages = torch.FloatTensor(advantages[batch])

                ratio = (log_probabilities - mb_old_log_probabilities.squeeze()).exp()
                min_adv = torch.where(mb_advantages > 0,
                    (1 + self.ppo_clip) * mb_advantages, (1 - self.ppo_clip) * mb_advantages)
                policy_loss = -(torch.min(ratio * mb_advantages, min_adv)).mean() - self.entropy_coefficient*entropy.mean()

                value_loss = (torch.FloatTensor(value_target[batch]) - predicted_values.squeeze()).pow(2).mean()
                self.top_layer_actor_critic.optimize(policy_loss, value_loss)
        adv += [value_target]
        return adv

from copy import deepcopy

def run(train_id=0):
    torch.set_num_threads(1)

    import gym
    env = gym.make("HalfCheetahHier-v2")

    agent_replay = ReplayMemory()

    # input = 17
    # output = 6
    top_spinal = SpinalActorCritic(SpinalNeuralNetwork([64, 64], obs_space=17*3+1, act_space=6+6))
    bottom_spinal = SpinalActorCritic(SpinalNeuralNetwork([64, 64], obs_space=17*3+6+6+1, act_space=6))

    spinal_network = SpinalHierarchicalNetwork(top_layer_ac=top_spinal,
        bottom_layer_ac=bottom_spinal, top_layer_freq=5, bottom_layer_freq=1,
        epochs=10, minibatch_size=500*15, timestep_size=3000*15, entropy_coefficient=0.0)

    timesteps = 0
    total_timesteps = 0
    max_timesteps = 30000000
    avg_action_magnitude = 0

    episode_itr = 0
    tr_avg_sum = 0.0
    avg_sum_rewards = 0.0

    saved_reward = list()
    saved_finish_mask = list()

    while total_timesteps < max_timesteps:
        game_over = False
        sensor_obs = env.reset()

        target_vel_x = 0

        freq_itr = 0
        freq_meta_itr = 0
        top_layer_action = np.zeros((1, 12))
        target_vel_x = 1.0 + torch.rand(1, 1) * 0.25
        while not game_over:
            if freq_itr == 0:
                top_layer_state = torch.cat((target_vel_x, torch.FloatTensor(sensor_obs).unsqueeze(0)), dim=1)
                top_layer_action, agent_replay = spinal_network(
                    x=top_layer_state, memory=agent_replay, top=True)
                agent_replay.sensor_states_top.append(deepcopy(top_layer_state.detach().numpy()[0]))
                agent_replay.actions_top.append(top_layer_action)
                clip_max = torch.FloatTensor([[1.05,  .785, .785, .7,  .87, .5,   10,  10,  10,  10,  10,  10]]) # this is both min and max, just get min and max separate
                clip_min = torch.FloatTensor([[-.52, -.785,  -.4, -1, -1.2, -.5,  -10, -10, -10, -10, -10, -10]])
                top_layer_action = np.clip(top_layer_action.clone(), a_max=clip_max, a_min=clip_min)
                agent_replay.reset_flags_top.append(1)

                # DO TARGET DELTA POSITION AND DELTA VELOCITY

            bottom_state = torch.cat((target_vel_x, top_layer_action, torch.FloatTensor(sensor_obs).unsqueeze(0)), dim=1)
            local_action, agent_replay = spinal_network(x=bottom_state, memory=agent_replay, bottom=True)

            agent_replay.sensor_states_bottom.append(deepcopy(bottom_state.detach().numpy()[0]))

            agent_replay.actions_bottom.append(local_action)

            local_action = local_action.squeeze(dim=1).numpy()
            sensor_obs, tr_rew, game_over, information = env.step(np.clip(local_action, a_min=-1, a_max=1))

            agent_replay.reset_flags_bottom.append(0 if game_over else 1)

            if game_over or freq_itr == 4:
                _vel_err = -1*(abs(target_vel_x.numpy()[0][0] - env.env.data.body_xvelp[1][0]))
                agent_replay.rewards_top.append(_vel_err + 4)
                if game_over:
                    agent_replay.reset_flags_top[-1] = 0

            reward = -(np.sum(np.abs(top_layer_action.flatten().numpy()[:6] - env.env.data.qpos[3:]))) + 4
            agent_replay.rewards_bottom.append(reward)

            avg_sum_rewards += reward
            tr_avg_sum += tr_rew

            timesteps += 1
            total_timesteps += 1
            freq_itr = (freq_itr+1)%spinal_network.top_layer_freq

        episode_itr += 1

        if timesteps > spinal_network.timestep_size:
            updates = spinal_network.learn(memory=agent_replay)
            #upd_list += updates
            avg_action_magnitude /= timesteps

            print("Time: {}, Bottom: {}, Top: {}, True: {}, Timestep: {}, ".format(
                round(timesteps/episode_itr, 5), round(sum(updates[0])/len(agent_replay.rewards_bottom), 56),
                 round(sum(updates[1])/len(agent_replay.rewards_top), 5), round(tr_avg_sum/episode_itr, 5), total_timesteps), end="")
            print("Target Vel: {}, Actual Vel, {}"
                  .format(round(target_vel_x.numpy()[0][0], 5), round(env.env.data.body_xvelp[1][0], 5)))

            timesteps = 0
            episode_itr = 0
            tr_avg_sum = 0.0
            avg_sum_rewards = 0.0
            avg_action_magnitude  = 0


            #with open("saved_model_{}_{}.pkl".format(net_type, train_id), "wb") as f:
            #    pickle.dump(agent, f)

            #with open("saved_weightupd_rew.pkl", "wb") as f:
            #    pickle.dump(upd_list, f)

            agent_replay.clear()
            saved_reward.clear()
            saved_finish_mask.clear()



run(1)






















