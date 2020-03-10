import torch
import numpy as np
import torch.nn as nn

from copy import deepcopy
import torch.optim as optim
from RodentNavigation.Networks.network_modules_torch import NetworkConnectivityModule


# plastic LSTM, with neuromod take in (x_n, v_t - v_t-1)
# train using recurrent SAC, burn in period
# modulate lower level hierarchy with weighted residuals

# H1:
#   1) top level CNN CORnet-s: take in first person images -> place and head direction
#   2) bottom level plastic LSTM: take in noisy velocity, CNN input -> target delta positions
#
#   **) output target delta position AND modulate motor cortex for
#     fine movements over/under obstacles (spatially oriented movement)
#
#   @) Dropout, ... constraints given from deepmind and other paper
#

# H2:
#   1) top level plastic LSTM: input target delta position -> target velocity and rotation
#
#   **) output target velocity and rotation AND modulate spinal neurons for fine movements (robust walking)
#
#   @) possibly use dropout and deepmind & other constraints
#

# H3:
#   1) top level linear ANN: input target velocity and rotation -> joint torques
#
#   **) output joint torques
#
#   @) possibly use dropout and deepmind & other constraints
#


class ValueModule(nn.Module):
    def __init__(self, inp_size, outp_size, activation=None, clip=1):
        """
        Network Sub-Module responsible for processing
         and connecting information between two layers
        """
        super(ValueModule, self).__init__()
        self.clip = clip
        self.inp_size = inp_size
        self.outp_size = outp_size
        self.activation = activation

        self.recurrent_trace = torch.zeros(
            1, self.outp_size, requires_grad=False)
        self.hebbian_trace = torch.zeros(
            (self.inp_size, self.outp_size), requires_grad=False)
        self.eligibility_trace = torch.zeros(
            (self.inp_size, self.outp_size), requires_grad=False)

        self.modulation_fan_in = nn.Linear(self.outp_size + 1, 1)
        self.modulation_fan_out = nn.Linear(1, self.outp_size)
        self.eligibility_eta = nn.Parameter(torch.ones(1), requires_grad=True)
        self.alpha_plasticity = nn.Parameter(torch.ones(1), requires_grad=True)

        self.value_predictor = nn.Linear(outp_size, 1)

        self.recurrent_layer = nn.Linear(self.outp_size, self.outp_size)

        self.layer = nn.Linear(self.inp_size, self.outp_size)

    def reset(self):
        self.hebbian_trace = self.hebbian_trace.detach() * 0
        self.recurrent_trace = self.recurrent_trace.detach() * 0
        self.eligibility_trace = self.eligibility_trace.detach() * 0
        
    def update_trace(self, pre_synaptic, post_synaptic, value):
        modulator = torch.cat((post_synaptic, value), dim=1)
        modulatory_signal = self.modulation_fan_out(
            torch.tanh(self.modulation_fan_in(modulator)))
        self.hebbian_trace = torch.clamp(
            self.hebbian_trace + modulatory_signal * self.eligibility_trace, max=self.clip, min=self.clip * -1)
        self.eligibility_trace = (torch.ones(1) - self.eligibility_eta)*\
            self.eligibility_trace + self.eligibility_eta*(torch.mm(pre_synaptic.t(), post_synaptic))

    def forward(self, x):
        pre_synaptic = x
        fixed_weights = self.layer(x)
        plastic_weights = x.mm(self.alpha_plasticity * self.hebbian_trace)
        post_synaptic = fixed_weights + plastic_weights
        if self.activation is not None:
            post_synaptic = self.activation(post_synaptic)
        value_pred = self.value_predictor(post_synaptic)
        self.update_trace(pre_synaptic=pre_synaptic, post_synaptic=post_synaptic, value=value_pred.detach())

        return post_synaptic, value_pred


class ValueModulatedNetwork(nn.Module):
    """
    Value-Modulated Recurrent neural network
    """
    def __init__(self):
        super(ValueModulatedNetwork, self).__init__()
        self.ff1 = ValueModule(inp_size=17, outp_size=64, activation=torch.tanh)
        self.ff2 = ValueModule(inp_size=64, outp_size=2)

    def forward(self, x):
        x, val_1 = self.ff1(x)
        x, val_2 = self.ff2(x)
        return x, val_2, val_1

    def learn(self, rewards, states, actions):
        pass


# todo: weight init orthogonal, clip gradient at 7.0

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from numpy import random
import torch.nn.functional as F
import random
import pickle
import time
import platform


np.set_printoptions(precision=4)

ADDINPUT = 4  # 1 inputs for the previous reward, 1 inputs for numstep, 1 unused,  1 "Bias" inputs


def train(paramdict):
    cuesshownall = []
    rewardsprevstepall = []
    modulatorall = []

    for numrun in range(4):

        print("Starting training...")
        params = {}
        # params.update(defaultParams)
        params.update(paramdict)

        #params['nbiter'] = 1
        #params['bs'] = 1

        print("Used params: ", params)
        print(platform.uname()) 
        NBINPUTBITS = params['cs'] + 1  # The additional bit is for the response cue (i.e. the "Go" cue)
        params['outputsize'] = 2  # "response" and "no response"
        params['inputsize'] = NBINPUTBITS + params[
            'outputsize'] + ADDINPUT  

        BS = params['bs']

        # Initialize random seeds (first two redundant?)
        print("Setting random seeds")
        np.random.seed(params['rngseed'])
        random.seed(params['rngseed'])
        torch.manual_seed(params['rngseed'])
        # print(click.get_current_context().params)

        net = ValueModulatedNetwork()

        print("Shape of all optimized parameters:", [x.size() for x in net.parameters()])
        allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
        print("Size (numel) of all optimized elements:", allsizes)
        print("Total size (numel) of all optimized elements:", sum(allsizes))

        # total_loss = 0.0
        print("Initializing optimizer")


        all_losses = []
        all_grad_norms = []
        all_losses_objective = []
        all_total_rewards = []
        lossbetweensaves = 0
        nowtime = time.time()
        totalnbtrials = 0
        nbtrialswithcc = 0

        print("Starting episodes!")

        for numepisode in range(params['nbiter']):

            loss = 0
            lossv = 0

            cuedata = []
            for nb in range(BS):
                cuedata.append([])
                for ncue in range(params['ni']):
                    assert len(cuedata[nb]) == ncue
                    foundsame = 1
                    cpt = 0
                    while foundsame > 0:
                        cpt += 1
                        if cpt > 10000:
                            # This should only occur with very weird parameters, e.g. cs=2, ni>4
                            raise ValueError("Could not generate a full list of different cues")
                        foundsame = 0
                        candidate = np.random.randint(2, size=params['cs']) * 2 - 1
                        for backtrace in range(ncue):
                            if np.array_equal(cuedata[nb][backtrace], candidate):
                                foundsame = 1

                    cuedata[nb].append(candidate)

            reward = np.zeros(BS)
            sumreward = np.zeros(BS)
            rewards = []
            vs = []
            vs2 = []
            logprobs = []
            cues = []
            for nb in range(BS):
                cues.append([])
            numactionschosen = np.zeros(BS, dtype='int32')

            nbtrials = np.zeros(BS)
            thistrialhascorrectcue = np.zeros(BS)
            triallength = np.zeros(BS, dtype='int32')
            correctcue = np.random.randint(params['ni'], size=BS)

            trialstep = np.zeros(BS, dtype='int32')

            cuesshown0 = []
            rewardsprevstep0 = []

            for numstep in range(params['eplen']):

                inputs = np.zeros((BS, params['inputsize']), dtype='float32')

                for nb in range(BS):

                    if trialstep[nb] == 0:
                        thistrialhascorrectcue[nb] = 0
                        triallength[nb] = params['ni'] // 2 + 3 + np.random.randint(params['ni'])
                        mycues = [x for x in range(params['ni'])]
                        random.shuffle(mycues)
                        mycues = mycues[:len(mycues) // 2]
                        for nc in range(triallength[nb] - 3 - len(mycues)):
                            mycues.append(-1)
                        random.shuffle(mycues)
                        mycues.insert(0, -1)
                        mycues.append(params['ni'])
                        mycues.append(-1)
                        assert (len(mycues) == triallength[nb])
                        cues[nb] = mycues

                    inputs[nb, :NBINPUTBITS] = 0
                    if -1 < cues[nb][trialstep[nb]] < params['ni']:
                        inputs[nb, :NBINPUTBITS - 1] = cuedata[nb][cues[nb][trialstep[nb]]][:]
                        if cues[nb][trialstep[nb]] == correctcue[nb]:
                            thistrialhascorrectcue[nb] = 1
                    if cues[nb][trialstep[nb]] == params['ni']:
                        inputs[nb, NBINPUTBITS - 1] = 1  # "Go" cue

                    inputs[nb, NBINPUTBITS + 0] = 1.0  # Bias neuron, probably not necessary
                    inputs[nb, NBINPUTBITS + 1] = numstep / params['eplen']
                    inputs[nb, NBINPUTBITS + 2] = 1.0 * reward[nb]  # Reward from previous time step
                    if numstep > 0:
                        inputs[nb, NBINPUTBITS + ADDINPUT + numactionschosen[nb]] = 1  # Previously chosen action

                inputsC = torch.from_numpy(inputs)


                y, v, v2 = net(torch.FloatTensor(inputsC))

                y = F.softmax(y, dim=1)
                # Must convert y to probas to use this !
                distrib = torch.distributions.Categorical(y)
                actionschosen = distrib.sample()
                logprobs.append(distrib.log_prob(actionschosen))
                numactionschosen = actionschosen.data.cpu().numpy()  # Turn to scalar
                cuesshown0.append(cues[0][trialstep[0]])
                rewardsprevstep0.append(float(reward[0]))

                reward = np.zeros(BS, dtype='float32')

                for nb in range(BS):
                    if numactionschosen[nb] == 1:
                        # Small penalty for any non-rest action taken
                        reward[nb] -= params['wp']

                    trialstep[nb] += 1
                    if trialstep[nb] == triallength[nb] - 1:
                        # This was the next-to-last step of the trial (and we showed the response signal, unless it was the first few steps in episode). 
                        assert (cues[nb][trialstep[nb] - 1] == params['ni'] or numstep < 2)
                        # We must deliver reward (which will be perceived by the agent at the next step), positive or negative, depending on response
                        if thistrialhascorrectcue[nb] and numactionschosen[nb] == 1:
                            reward[nb] += params['rew']
                        elif (not thistrialhascorrectcue[nb]) and numactionschosen[nb] == 0:
                            reward[nb] += params['rew']
                        else:
                            reward[nb] -= params['rew']

                        if np.random.rand() < params['pf']:
                            reward[nb] = -reward[nb]

                    if trialstep[nb] == triallength[nb]:
                        # This was the last step of the trial (and we showed no input)
                        assert (cues[nb][trialstep[nb] - 1] == -1 or numstep < 2)
                        nbtrials[nb] += 1
                        totalnbtrials += 1
                        if thistrialhascorrectcue[nb]:
                            nbtrialswithcc += 1
                            # nbrewardabletrials += 1 
                        # Trial is dead, long live trial
                        trialstep[nb] = 0
                rewards.append(reward)
                vs.append(v)
                vs2.append(v2)
                sumreward += reward

                loss += (params['bent'] * y.pow(2).sum() / BS)
            R = Variable(torch.zeros(BS), requires_grad=False)
            gammaR = params['gr']
            for numstepb in reversed(range(params['eplen'])):
                R = gammaR * R + Variable(torch.from_numpy(rewards[numstepb]), requires_grad=False)
                ctrR = R - vs[numstepb][0]
                # todo: lossv += (R - vs2[numstepb][0]).pow(2).sum() / BS
                lossv += ctrR.pow(2).sum() / BS
                loss -= (logprobs[numstepb] * ctrR.detach()).sum() / BS  # Need to check if detach() is OK
                # pdb.set_trace()
            loss += params['blossv'] * lossv
            loss /= params['eplen']

            lossnum = float(loss)
            lossbetweensaves += lossnum
            all_losses_objective.append(lossnum)
            all_total_rewards.append(sumreward.mean())

            if (numepisode + 1) % params['pe'] == 0:
                print(numepisode, "====")
                #print("Mean loss: ", lossbetweensaves / params['pe'])
                lossbetweensaves = 0
                print("Mean reward: ", np.sum(all_total_rewards[-params['pe']:]) / params['pe'])
                previoustime = nowtime
                nowtime = time.time()
                #print("Time spent on last", params['pe'], "iters: ", nowtime - previoustime)

            if (numepisode + 1) % params['save_every'] == 0:
                #print("Saving files...")
                losslast100 = np.mean(all_losses_objective[-100:])
                print("Average loss over the last 100 episodes:", losslast100)


if __name__ == "__main__":
    # defaultParams = {
    #    'type' : 'lstm',
    #    'seqlen' : 200,
    #    'hs': 500,
    #    'activ': 'tanh',
    #    'steplr': 10e9,  # By default, no change in the learning rate
    #    'gamma': .5,  # The annealing factor of learning rate decay for Adam
    #    'imagesize': 31,
    #    'nbiter': 30000,
    #    'lr': 1e-4,
    #    'test_every': 10,
    #    'save_every': 3000,
    #    'rngseed':0
    # }

    parser = argparse.ArgumentParser()
    parser.add_argument("--rngseed", type=int, help="random seed", default=0)
    parser.add_argument("--rew", type=float,
                        help="reward value (reward increment for taking correct action after correct stimulus)",
                        default=1.0)
    parser.add_argument("--wp", type=float, help="penalty for hitting walls", default=.0)
    parser.add_argument("--bent", type=float,
                        help="coefficient for the entropy reward (really Simpson index concentration measure)",
                        default=0.03)
    parser.add_argument("--blossv", type=float, help="coefficient for value prediction loss", default=.1)
    parser.add_argument("--bv", type=float, help="coefficient for value prediction loss", default=.1)
    parser.add_argument("--alg", help="meta-learning algorithm (A3C or REI or REIE or REIT)", default='REIT')
    parser.add_argument("--rule", help="learning rule ('hebb' or 'oja')", default='hebb')
    parser.add_argument("--type", help="network type ('lstm' or 'rnn' or 'plastic')", default='modul')
    parser.add_argument("--da", help="transformation function of DA signal (tanh or sig or lin)", default='tanh')
    parser.add_argument("--gr", type=float, help="gammaR: discounting factor for rewards", default=.9)
    parser.add_argument("--lr", type=float, help="learning rate (Adam optimizer)", default=1e-4)
    parser.add_argument("--fm", type=int,
                        help="if using neuromodulation, do we modulate the whole network (1) or just half (0) ?",
                        default=1)
    parser.add_argument("--ni", type=int, help="number of different inputs", default=2)
    parser.add_argument("--nu", type=float, help="REINFORCE baseline time constant", default=.1)
    parser.add_argument("--addpw", type=int, help="are plastic weights purely additive (1) or forgetting (0) ?",
                        default=2)
    parser.add_argument("--clamp", type=int, help="inputs clamped (1), fully clamped (2) or through linear layer (0) ?",
                        default=0)
    parser.add_argument("--eplen", type=int, help="length of episodes", default=100)
    parser.add_argument("--hs", type=int, help="size of the recurrent (hidden) layer", default=100)
    parser.add_argument("--is", type=int, help="do we initialize hidden state after each trial (1) or not (0) ?",
                        default=0)
    parser.add_argument("--cs", type=int, help="cue size - number of bits for each cue", default=10)
    parser.add_argument("--pf", type=float, help="probability of flipping the reward (.5 = pure noise)", default=0)
    parser.add_argument("--l2", type=float, help="coefficient of L2 norm (weight decay)", default=1e-5)
    parser.add_argument("--bs", type=int, help="batch size", default=1)
    parser.add_argument("--gc", type=float, help="gradient clipping", default=1000.0)
    parser.add_argument("--eps", type=float, help="epsilon for Adam optimizer", default=1e-6)
    parser.add_argument("--nbiter", type=int, help="number of learning cycles", default=1000000)
    parser.add_argument("--save_every", type=int, help="number of cycles between successive save points", default=200)
    parser.add_argument("--pe", type=int,
                        help="'print every', number of cycles between successive printing of information", default=100)
    args = parser.parse_args()
    argvars = vars(args)
    argdict = {k: argvars[k] for k in argvars if argvars[k] != None}
    train(argdict)













