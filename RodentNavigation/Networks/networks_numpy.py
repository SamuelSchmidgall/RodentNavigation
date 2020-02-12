import numpy as np
from copy import deepcopy
from RodentNavigation.Networks.network_modules_numpy import NetworkModule

def identity(x):
    return x


class ESOptimizer:
    def __init__(self, sample_type, num_eps_samples, noise_std=0.01):
        self.noise_std = noise_std
        self.sample_type = sample_type
        self.num_eps_samples = num_eps_samples
        if self.sample_type == "antithetic":
            assert (self.num_eps_samples % 2 == 0), "Population size must be even"
            self.half_popsize = int(self.num_eps_samples / 2)

    def sample(self, params, seed):
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
    def __init__(self, input_size, output_size, upstream_dim, num_eps_samples=64):
        self.params = list()
        self.input_size = input_size
        self.output_size = output_size
        self.input_upstream_dim = upstream_dim*2
        self.es_optim = ESOptimizer("antithetic", num_eps_samples)

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
        self.layer_1_prev = self.layer_1_prev * 0
        self.layer_2_prev = self.layer_2_prev * 0
        self.layer_3_prev = self.layer_3_prev * 0
        # todo: reset each module

    def parameters(self):
        params = list()
        for _param in range(len(self.params)):
            params.append(self.params[_param].params())
        return np.concatenate(params, axis=0)

    def generate_eps_samples(self, seed=None):
        params = self.parameters()
        sample = self.es_optim.sample(params, seed)
        return sample

    def update_params(self, eps_sample):
        param_itr = 0
        for _param in range(len(self.params)):
            pre_param_itr = param_itr
            param_itr += self.params[_param].parameters.size
            param_sample = eps_sample[pre_param_itr:param_itr]
            self.params[_param].update_params(param_sample)

    def forward(self, x):
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




upstream_bottleneck = 16
spinal_net = SpinalNetworkES(32, 34, upstream_bottleneck)
random_inp = np.ones((1, 32))
spinal_net.forward(random_inp)
samples = spinal_net.generate_eps_samples()
spinal_net.update_params(samples[0])























