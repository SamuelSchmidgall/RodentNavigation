import numpy as np
from copy import deepcopy
from RodentNavigation.Networks.utils import *
from RodentNavigation.Networks.network_modules_numpy import NetworkModule


class ParamSampler:
    def __init__(self, sample_type, num_eps_samples, noise_std=0.005):
        """
        Evolutionary Strategies Optimizer
        :param sample_type: (str) type of noise sampling
        :param num_eps_samples: (int) number of noise samples to generate
        :param noise_std: (float) nosie standard deviation
        """
        self.noise_std = noise_std
        self.sample_type = sample_type
        self.num_eps_samples = num_eps_samples
        if self.sample_type == "antithetic":
            assert (self.num_eps_samples % 2 == 0), "Population size must be even"
            self.half_popsize = int(self.num_eps_samples / 2)

    def sample(self, params, seed):
        """
        Sample noise for network parameters
        :param params: (ndarray) network parameters
        :param seed: (int) random seed used to sample
        :return: (ndarray) sampled noise
        """
        if seed is not None:
            rand_m = np.random.RandomState(seed)
        else:
            rand_m = np.random.RandomState()

        sample = None
        if self.sample_type == "antithetic":
            epsilon_half = rand_m.randn(self.half_popsize, params.size)
            sample = np.concatenate([epsilon_half, - epsilon_half]) * self.noise_std
        elif self.sample_type == "normal":
            sample = rand_m.randn(self.num_eps_samples, params.size) * self.noise_std

        return sample


class SpinalNetworkES:
    def __init__(self, input_size, output_size, upstream_dim, num_eps_samples=64, sample_type="antithetic"):
        """
        Spinal Network with lateral, residual upstream, residual downstream,
         recurrence and eligibility modulated weights for each set of connections
        :param input_size: (int) size of observation space
        :param output_size: (int) size of action space
        :param upstream_dim: (int) upsteam bottleneck dimensionality
        :param num_eps_samples: (int) number of epsilon samples (population size)
        :param sample_type: (str) network noise sampling type
        """
        self.params = list()
        self.input_size = input_size
        self.output_size = output_size
        self.input_upstream_dim = upstream_dim*2
        self.es_optim = ParamSampler(sample_type=sample_type, num_eps_samples=num_eps_samples)

        recur_ff1_meta = {
            "clip":1, "activation": identity,
            "input_size": input_size+self.input_upstream_dim, "output_size": 64}
        self.recur_plastic_ff1 = \
            NetworkModule("eligibility_recurrent", recur_ff1_meta)
        self.params.append(self.recur_plastic_ff1)
        recur_ff2_meta = {
            "clip":1, "activation": identity, "input_size": 64, "output_size": 64}
        self.recur_plastic_ff2 = \
            NetworkModule("eligibility_recurrent", recur_ff2_meta)
        self.params.append(self.recur_plastic_ff2)
        recur_ff3_meta = {
            "clip":1, "activation": identity, "input_size": 64, "output_size": output_size}
        self.recur_plastic_ff3 = \
            NetworkModule("eligibility_recurrent", recur_ff3_meta)
        self.params.append(self.recur_plastic_ff3)

        resid_down_inp_2_meta = {
            "clip":1, "activation": identity,
            "input_size": input_size+self.input_upstream_dim, "output_size": 64}
        self.resid_downstream_inp_2 = \
            NetworkModule("eligibility", resid_down_inp_2_meta)
        self.params.append(self.resid_downstream_inp_2)
        resid_down_inp_3_meta = {
            "clip":1, "activation": identity,
            "input_size": input_size+self.input_upstream_dim, "output_size": output_size}
        self.resid_downstream_inp_3 = \
            NetworkModule("eligibility", resid_down_inp_3_meta)
        self.params.append(self.resid_downstream_inp_3)
        resid_down_1_3_meta = {
            "clip":1, "activation": identity, "input_size": 64, "output_size": output_size}
        self.resid_downstream_1_3 = \
            NetworkModule("eligibility", resid_down_1_3_meta)
        self.params.append(self.resid_downstream_1_3)

        resid_up_2_inp_meta = {
            "clip":1, "activation": identity, "input_size": 64, "output_size": upstream_dim}
        self.resid_upstream_2_inp = \
            NetworkModule("eligibility", resid_up_2_inp_meta)
        self.params.append(self.resid_upstream_2_inp)
        resid_up_3_inp_meta = {
            "clip":1, "activation": identity, "input_size": output_size, "output_size": upstream_dim}
        self.resid_upstream_3_inp = \
            NetworkModule("eligibility", resid_up_3_inp_meta)
        self.params.append(self.resid_upstream_3_inp)
        resid_up_3_1_meta = {
            "clip":1, "activation": identity, "input_size": output_size, "output_size": 64}
        self.resid_upstream_3_1 = \
            NetworkModule("eligibility", resid_up_3_1_meta)
        self.params.append(self.resid_upstream_3_1)

        resid_lat_1_inp_meta = {
            "clip":1, "activation": identity, "input_size": 64, "output_size": 64}
        self.resid_lat_1 = \
            NetworkModule("eligibility", resid_lat_1_inp_meta)
        self.params.append(self.resid_lat_1)
        resid_lat_2_inp_meta = {
            "clip":1, "activation": identity, "input_size": 64, "output_size": 64}
        self.resid_lat_2 = \
            NetworkModule("eligibility", resid_lat_2_inp_meta)
        self.params.append(self.resid_lat_2)
        resid_lat_3_inp_meta = {
            "clip":1, "activation": identity, "input_size": output_size, "output_size": output_size}
        self.resid_lat_3 = \
            NetworkModule("eligibility", resid_lat_3_inp_meta)
        self.params.append(self.resid_lat_3)

        self.layer_1_prev = np.zeros((1, recur_ff1_meta["output_size"]))
        self.layer_2_prev = np.zeros((1, recur_ff2_meta["output_size"]))
        self.layer_3_prev = np.zeros((1, recur_ff3_meta["output_size"]))

    def reset(self):
        """
        Reset inter-lifetime network parameters
        :return: None
        """
        for _param in self.params:
            _param.reset()
        self.layer_1_prev = self.layer_1_prev * 0
        self.layer_2_prev = self.layer_2_prev * 0
        self.layer_3_prev = self.layer_3_prev * 0

    def parameters(self):
        """
        Return list of network parameters
        :return: (ndarray) network parameters
        """
        params = list()
        for _param in range(len(self.params)):
            params.append(self.params[_param].params())
        return np.concatenate(params, axis=0)

    def generate_eps_samples(self, seed=None):
        """
        Generate noise samples for list of parameters
        :param seed: (int) random number seed
        :return: (ndarray) parameter noise
        """
        params = self.parameters()
        sample = self.es_optim.sample(params, seed)
        return sample

    def update_params(self, eps_sample):
        """
        Update internal network parameters
        :param eps_sample: (ndarray) noise sample
        :return: None
        """
        param_itr = 0
        for _param in range(len(self.params)):
            pre_param_itr = param_itr
            param_itr += self.params[_param].parameters.size
            param_sample = eps_sample[pre_param_itr:param_itr]
            self.params[_param].update_params(param_sample)

    def forward(self, x):
        """
        Forward propagate input value
        :param x: (ndarray) state input
        :return: (ndarray) post synaptic activity at final layer
        """
        upstr_res_inp_2 = self.resid_upstream_2_inp.forward(self.layer_2_prev)
        upstr_res_inp_3 = self.resid_upstream_3_inp.forward(self.layer_3_prev)
        x = np.concatenate([x, upstr_res_inp_2, upstr_res_inp_3], axis=1)

        pre_synaptic_ff1 = x
        lat_res1 = self.resid_lat_1.forward(self.layer_1_prev)
        upstr_res_3_1 = self.resid_upstream_3_1.forward(self.layer_3_prev)
        post_synaptic_ff1 = np.tanh(
            self.recur_plastic_ff1.forward(pre_synaptic_ff1) + upstr_res_3_1 + lat_res1)

        pre_synaptic_ff2 = post_synaptic_ff1
        lat_res2 = self.resid_lat_2.forward(self.layer_2_prev)
        downstr_res_inp_2 = self.resid_downstream_inp_2.forward(x)
        post_synaptic_ff2 = np.tanh(
            self.recur_plastic_ff2.forward(pre_synaptic_ff2) + downstr_res_inp_2 + lat_res2)

        pre_synaptic_ff3 = post_synaptic_ff2
        lat_res3 = self.resid_lat_3.forward(self.layer_3_prev)
        downstr_res_inp_3 = self.resid_downstream_inp_3.forward(x)
        downstr_res_1_3 = self.resid_downstream_1_3.forward(post_synaptic_ff1)
        post_synaptic_ff3 = \
            self.recur_plastic_ff3.forward(pre_synaptic_ff3) + downstr_res_1_3 + downstr_res_inp_3 + lat_res3

        self.layer_1_prev = post_synaptic_ff1
        self.layer_2_prev = post_synaptic_ff2
        self.layer_3_prev = post_synaptic_ff3

        return post_synaptic_ff3




class EvolutionaryOptimizer:
    def __init__(self, network, num_workers=4, epsilon_samples=64):
        assert (epsilon_samples % num_workers == 0), "Epsilon sample size not divis num workers"
        self.network = network
        self.num_workers = num_workers
        self.epsilon_samples = epsilon_samples

    def parallel_returns(self, x):
        worker_id, seed = x
        return compute_hebb_returns(seed, environment, self.network, self.epsilon_samples//self.num_workers)

    def update(self, iteration):
        returns = list()
        random_seeds = [iteration*self.num_workers + _ for _ in range(self.num_workers)]
        for _worker in range(self.num_workers):
            seed = random_seeds[_worker]
            returns, timesteps = values[_worker]

            #timestep_list.append(timesteps)
            #total_timesteps += timesteps
            #sample_returns += returns.tolist()
            #samples += [self.nes.hebb_sample(seed)]


upstream_bottleneck = 16
spinal_net = SpinalNetworkES(32, 34, upstream_bottleneck)
random_inp = np.ones((1, 32))
spinal_net.forward(random_inp)
samples = spinal_net.generate_eps_samples()
spinal_net.update_params(samples[0])
es_optim = EvolutionaryOptimizer(spinal_net)


def compute_hebb_returns(seed, environment, network, eps_samples, num_env_rollouts=1):
    avg_stand = 0
    returns = list()
    local_env = environment
    total_env_interacts = 0
    max_env_interacts = 1000
    network_cpy = deepcopy(network)

    # iterate through the noise samples
    for _sample in range(eps_samples):
        return_avg = 0.0
        network = deepcopy(network_cpy)

        # iterate through the target number of env rollouts
        for _roll in range(num_env_rollouts):
            network.reset()
            state = local_env.reset()
            # run the simulation until it terminates or max env interation terminates it
            for _inter in range(max_env_interacts):
                # forward propagate using noisy weights and plasticity values, also update trace
                action = network.hebb_forward(state)[0]
                action = np.clip(action, a_min=-1, a_max=1)
                # interact with environment
                state, reward, game_over, _ = local_env.step(action)
                return_avg += reward
                # end sim iteration if termination state reached
                if game_over:
                    break
                avg_stand += 1
                total_env_interacts += 1
        returns.append(return_avg / num_env_rollouts)
    return np.array(returns), total_env_interacts

















