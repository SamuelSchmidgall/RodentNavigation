import math
import torch
import numpy as np
from torch import nn
from copy import deepcopy
import torch.optim as optim
import torch.nn.functional as F


def _uniform_init(tensor, param=3e-3):
    return tensor.data.uniform_(-param, param)

def layer_init(layer, weight_init = _uniform_init, bias_init = _uniform_init ):
    weight_init(layer.weight)
    bias_init(layer.bias)

def uniform_init(layer):
    layer_init(layer, weight_init = _uniform_init, bias_init = _uniform_init )


class ReplayMemory:
    def __init__(self, max_size=None):
        self.actions = list()
        self.rewards = list()
        self.commands = list()
        self.log_probs = list()
        self.reset_flags = list()
        self.sensor_states = list()
        self.visual_states = list()
        self.next_commands = list()
        self.next_image_batch = list()
        self.next_sensor_states = list()
        self.state_action_pairs = list()

    def clear(self):
        self.actions = list()
        self.rewards = list()
        self.commands = list()
        self.log_probs = list()
        self.reset_flags = list()
        self.next_commands = list()
        self.sensor_states = list()
        self.visual_states = list()
        self.next_image_batch = list()
        self.next_sensor_states = list()
        self.state_action_pairs = list()



class PlasticityModule(nn.Module):
    def __init__(self, dim_1, dim_2, plasticity=True, plasticity_type="neuromodulated_linear"):
        super(PlasticityModule, self).__init__()
        self.COMPRESS_DIM = 3
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

class NetworkModule(nn.Module):
    def __init__(self, dim_1, dim_2, module_type='continuous_reinforcement', optional_args=None):
        super(NetworkModule, self).__init__()

        self.dimension_1 = dim_1
        self.dimension_2 = dim_2
        self.module_type = module_type

        if module_type not in self.module_types:
            raise Exception("{} is not a module type".format(module_type))

        if module_type == "continuous_reinforcement":
            # linear feedforward
            self.activation = torch.tanh
            self.linear = nn.Linear(dim_1, dim_2)
            uniform_init(self.linear)

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
            uniform_init(self.linear)

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
            uniform_init(self.linear)
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


class NeuralNetwork(nn.Module):
    def __init__(self, latent_shape, environment=None, act_space=None, obs_space=None,
                 plasticity=None, network_type="cont_pg_rl", module_arguments=None):
        super(NeuralNetwork, self).__init__()

        self.environment = environment
        self.network_type = network_type
        if environment is not None:
            ac_space = environment.action_space.shape[0]
        else:
            ac_space = act_space.shape[0]

        if environment is not None:
            ob_space = environment.observation_space.shape[0]
        else:
            ob_space = obs_space

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
                NetworkModule(dim_1=self.network_shape[_],
                    dim_2=self.network_shape[_ + 1], module_type="continuous_reinforcement",
                    optional_args=module_arguments[_]) for _ in range(len(self.network_shape) - 2)]
            self.network_modules.append(NetworkModule(dim_1=self.network_shape[-2], dim_2=self.network_shape[-1],
                    module_type="continuous_reinforcement_final", optional_args=module_arguments[-1]))

        elif network_type == "vanilla":
            """ Continuous policy gradient reinforcement learning """
            self.network_modules = [
                NetworkModule(dim_1=self.network_shape[_],
                    dim_2=self.network_shape[_ + 1], module_type="continuous_reinforcement",
                    optional_args=module_arguments[_]) for _ in range(len(self.network_shape) - 1)]
            self.network_modules[-1].activation = nn.Identity()

        elif network_type == "vision":
            """ Continuous policy gradient reinforcement learning """
            self.network_modules = [
                NetworkModule(dim_1=self.network_shape[_],
                    dim_2=self.network_shape[_ + 1], module_type="continuous_reinforcement_vision",
                    optional_args=module_arguments[_]) for _ in range(len(self.network_shape) - 2)]
            self.network_modules.append(NetworkModule(dim_1=self.network_shape[-2], dim_2=self.network_shape[-1],
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
            value_modules.append(NetworkModule(dim_1=_module.dimension_1, dim_2=_module.dimension_2,
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

class MultiSensoryNeuralNetwork(nn.Module):
    def __init__(self, sensor_network, visual_network, environment, flatten_dim, input_dim, network_type="cont_pg_rl"):
        super(MultiSensoryNeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.flatten_dim = flatten_dim
        self.environment = environment
        self.network_type = network_type
        self.sensor_network = sensor_network
        self.visual_network = visual_network

        self.flatten = nn.Linear(flatten_dim, input_dim)

        if self.network_type == "cont_pg_rl":
            self.final_module = NetworkModule(input_dim,
                environment.action_space.shape[0], module_type="continuous_reinforcement_final")
        elif self.network_type == "value":
            self.final_module = NetworkModule(input_dim,
                environment.action_space.shape[0], module_type="value_module_final")

    def forward(self, sensor, img):
        state_1 = self.sensor_network(sensor)
        state_2 = self.visual_network(img)

        if state_1.shape != state_2.shape:
            raise TypeError("Multi-Sensory incompatible shapes")

        multisensory_state = torch.tanh(state_1 + state_2)
        return self.final_module(multisensory_state)

    def params(self):
        sensor_net_params = self.sensor_network.params()
        vision_net_params = self.visual_network.params()
        final_module_params = list(self.final_module.parameters())
        return sensor_net_params+vision_net_params+final_module_params

    def generate_value_function(self):
        # copy network topology and change final module
        network = deepcopy(self)
        network.network_type = "value"
        network.final_module = NetworkModule(network.input_dim,
            network.environment.action_space.shape[0], module_type="value_module_final")
        return network

class ActorCritic(nn.Module):
    def __init__(self, policy, net_type="linear",
            policy_learning_rate=0.0003, value_learning_rate=0.0003, optimizer=optim.Adam):

        super(ActorCritic, self).__init__()
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



