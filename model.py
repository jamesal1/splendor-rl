import torch
from torch import nn
import torch.nn.functional as F
from constants import *
if cuda_on:
    cuda_device = torch.device("cuda")
else:
    cuda_device = torch.device("cpu")

if half_precision:
    precision = torch.float16
else:
    precision = torch.float32

import time

from torch.nn.parameter import Parameter
import random
import time

input_size = 401
aux_size = 90 + 25
action_size = 214
discard_size = 18
noble_size = 10

take_info_cuda = torch.tensor(TAKE_INFO, device=cuda_device)
take_cuda = take_info_cuda[TAKE_START:PASS,:]
card_info_cuda = torch.tensor(CARD_INFO, device=cuda_device)
card_cost_cuda = card_info_cuda[:,COST:COST+5]
noble_info_cuda = torch.tensor(NOBLE_INFO, device=cuda_device)
def resource_mask(states):
    resources = states[:,VIEW_PLAYER+CHIPS:VIEW_PLAYER+CHIPS+5]+states[:,VIEW_PLAYER+CARDS:VIEW_PLAYER+CARDS+5]
    shortfall = card_cost_cuda.view(1,90,5) - resources.view(-1,1,5)
    return (shortfall * (shortfall > 0)).sum(axis=2) <= states[:,VIEW_PLAYER+GOLD].view(-1,1)



class CubeAct(nn.Module):


    def forward(self,x):

        return x ** 3


class PerturbedLinear(nn.Linear):

    def __init__(self, in_features, out_features, directions, perturbed_flag, bias=True):
        super(PerturbedLinear, self).__init__(in_features,out_features,bias)
        self.directions = directions
        self.in_features = in_features
        self.out_features = out_features
        self.perturbed_weight = None
        if bias:
            self.perturbed_bias = None
        self.perturbed_flag = perturbed_flag
        self.set_seed()
        self.neg_mask = None
        self.noise_scale = None

    def set_seed(self, seed=None):
        self.seed = seed if seed is not None else random.randrange(100000000)

    def set_noise_scale(self, noise_scale):
        self.noise_scale = noise_scale

    def set_noise(self):
        if self.perturbed_weight is None:
            gen = torch.cuda.manual_seed(self.seed)
            self.perturbed_weight = torch.zeros(self.directions, self.out_features, self.in_features, device=cuda_device, dtype=precision)
            self.perturbed_weight.normal_(std=self.noise_scale, generator=gen)
            if self.bias is not None and self.perturbed_bias is None:
                self.perturbed_bias = torch.zeros(self.directions, self.out_features, device=cuda_device, dtype=precision)
                self.perturbed_bias.normal_(std=self.noise_scale, generator=gen)




    def set_grad(self, weights):
        if half_precision:
            weights = weights.half()
        self.set_noise()
        self.weight.grad = (self.perturbed_weight * weights.view(self.directions, 1, 1)).sum(dim=0)
        if self.bias is not None:
            self.bias.grad = (self.perturbed_bias * weights.view(self.directions, 1)).sum(dim=0)
        self.clear_noise()


    def clear_noise(self):
        self.perturbed_weight = None
        if self.bias is not None:
            self.perturbed_bias = None
        # self.neg_mask = None

    def forward(self, input):
        start = time.time()
        unperturbed = F.linear(input, self.weight, self.bias)
        # print("unperturbed",self.perturbed_flag[0], time.time() - start)
        if self.perturbed_flag[0]:
            start = time.time()
            self.set_noise()
            # print("noise", time.time() - start)
            batch_view_input = input.view(self.directions, -1, self.in_features)
            repeat_size = batch_view_input.size(1)
            start = time.time()
            if self.bias is not None:
                perturbations = torch.baddbmm(self.perturbed_bias.view(self.directions, 1, self.out_features),
                                              batch_view_input,
                                              self.perturbed_weight.permute([0, 2, 1]))
            else:
                perturbations = torch.bmm(batch_view_input, self.perturbed_weight.permute([0, 2, 1]))
            # print("perturbed", time.time() - start)
            # start = time.time()
            if self.neg_mask is None or self.neg_mask.size(1)!=repeat_size:
                self.neg_mask = torch.ones((1,repeat_size,1), device=cuda_device, dtype=precision)
                self.neg_mask[:, repeat_size // 2:, :] *= -1
            # print("negative", time.time() - start)
            # start = time.time()
            # add = (perturbations*self.neg_mask).view(*unperturbed.size()) + unperturbed
            add = torch.addcmul(unperturbed.view(*perturbations.size()),perturbations,self.neg_mask).view(*unperturbed.size())
            # print("add", time.time() - start)
            # self.clear_noise()
            return add
        return unperturbed


class Model(nn.Module):


    def __init__(self, **kwargs):
        super(Model,self).__init__()
        self.memory_size=kwargs.get("memory_size",4)
        self.perturbed_flag = [0]

    def get_action(self, input, memory, perturbed=False):
        self.perturbed_flag[0] = perturbed
        if half_precision:
            input = input.half()
        else:
            input = input.float()
        chip_sum = input[:,VIEW_PLAYER+CHIPS:VIEW_PLAYER+CHIPS+5] + input[:,VIEW_OTHER+CHIPS:VIEW_OTHER+CHIPS+5]
        reserve_mask = input[:, VIEW_PLAYER+RESERVED:VIEW_PLAYER+RESERVED+90]
        chip_mask = input[:, VIEW_PLAYER+CHIPS:VIEW_PLAYER+CHIPS+6] > 0

        purchase_mask = (input[:, VIEW_PLAY:VIEW_PLAY+90] + reserve_mask) * resource_mask(input)
        take_mask = (((chip_sum.view(-1,1,5) + take_cuda.view(1,PASS,5)) < 5).sum(axis=2) == 5)
        # if self.half_size:
        #     take_mask = take_mask.half()
        # else:
        #     take_mask = take_mask.float()
        aux = torch.cat((purchase_mask.bool(),take_mask),dim=1)
        action, discard, noble, memory = self.forward(input, aux, memory)
        action = (action - action.min()).view(-1,action_size)
        discard = (discard - discard.min()).view(-1, discard_size)
        noble = (noble - noble.min()).view(-1, noble_size)
        memory = memory.view(-1,self.memory_size)
        start = time.time()

        action[:, PURCHASE_START:PURCHASE_END] *= purchase_mask
        action[:, RESERVE_START:RESERVE_END] *= input[:, VIEW_PLAY:VIEW_PLAY+90]
        action[:, RESERVE_START:RESERVE_TOP_END] *= (input[:,VIEW_PLAYER+L1_RESERVED:VIEW_PLAYER+L1_RESERVED+3].sum(axis=1) < 3).view(-1,1)
        # print("card mask time", time.time() - start)
        start = time.time()

        action[:,TAKE_START:PASS] *= take_mask
        # action[:,PASS] *= chip_sum.sum(axis=1) == 20
        action[:,TAKE2_START:TAKE2_END] *= chip_sum == 0
        # print("take mask time", time.time() - start)

        noble_req = ((input[:,VIEW_PLAYER+CARDS:VIEW_PLAYER+CARDS+5].view(-1,1,5) - noble_info_cuda.view(1,10,5)) >= 0).sum(axis=2) == 5
        noble *= input[:,VIEW_NOBLES:VIEW_NOBLES + 10] * noble_req
        _, action = torch.max(action, dim=1)
        _, discard = torch.max(discard.view(-1, 3, 6) * chip_mask.view(-1, 1, 6), dim=2)
        _, noble = torch.max(noble, dim=1)
        return action, discard, noble, memory


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias.data.fill_(0.01)




class PerturbedModel():

    def __init__(self, model):

        self.model = model

    def get_action(self, input, memory):
        return self.model.get_action(input, memory, True)

    def __getattr__(self, name):
        def ret(*args):
            def helper(m):
                if type(m) == PerturbedLinear:
                    getattr(m,name)(*args)
            self.model.apply(helper)
        return ret



class DenseNet(Model):


    def __init__(self, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        layer_sizes = kwargs.get("layer_sizes", [1024])
        directions = kwargs.get("directions")
        layers = []
        s = input_size + self.memory_size + aux_size
        # s = input_size + self.memory_size

        for t in layer_sizes:
            layers += [PerturbedLinear(s,t, directions, self.perturbed_flag)]
            # layers += [nn.BatchNorm1d(t)]
            layers += [nn.ELU()]
            # layers += [CubeAct()]

            s = t
        self.layers = nn.Sequential(*layers)
        self.output_0 = PerturbedLinear(t, action_size, directions, self.perturbed_flag)
        self.output_1 = PerturbedLinear(t, discard_size, directions, self.perturbed_flag)
        self.output_2 = PerturbedLinear(t, noble_size, directions, self.perturbed_flag)
        self.output_memory = PerturbedLinear(t, self.memory_size, directions, self.perturbed_flag)
        self.apply(init_weights)




    def forward(self, input, aux, memory):
        if memory is None:
            memory = torch.zeros(input.size(0), self.memory_size, device=cuda_device)
        if self.output_0.weight.dtype == torch.float16:
            aug_input = torch.cat((input.half(), aux.half(), memory.half()), dim=1)
            # aug_input = torch.cat((input.half(), memory.half()), dim=1)
        else:
            aug_input = torch.cat((input.float(), aux.float(), memory), dim=1)
            # aug_input = torch.cat((input.float(), memory), dim=1)

        features = self.layers.forward(aug_input)
        action = self.output_0.forward(features)
        discard = self.output_1.forward(features)
        noble = self.output_2.forward(features)
        memory = torch.tanh(self.output_memory.forward(features))
        return action, discard, noble, memory

class TransformerNet(Model):
    def __init__(self, **kwargs):
        super(TransformerNet, self).__init__(**kwargs)

        self.transformer_input_size = kwargs.get("transformer_input_size", 64)
        self.transformer_length = kwargs.get("transformer_length", 64)
        self.matrix_size = self.transformer_input_size * self.transformer_length
        transformer_heads = kwargs.get("transformer_heads", 8)
        layer_sizes = kwargs.get("layer_sizes", [1024] * 3)
        layers = []
        s = input_size + self.memory_size + aux_size
        s = input_size + self.memory_size
        self.pre_transform = nn.Linear(s, self.transformer_input_size*self.transformer_length)
        layers += [nn.Tanh(inplace=True)]
        layers += [nn.TransformerEncoderLayer(self.transformer_input_size, transformer_heads,
                                                    dim_feedforward=512, dropout=0, activation='relu')]


        self.layers = nn.Sequential(*layers)
        self.output_0 = nn.Linear(self.matrix_size, action_size)
        self.output_1 = nn.Linear(self.matrix_size, discard_size)
        self.output_2 = nn.Linear(self.matrix_size, noble_size)
        self.output_memory = nn.Linear(self.matrix_size, self.memory_size)




    def forward(self, input, aux, memory):
        if memory is None:
            memory = torch.zeros(input.size(0), self.memory_size, device=cuda_device)
        if self.output_0.weight.dtype == torch.float16:
            # aug_input = torch.cat((input.half(), aux.half(), memory.half()), dim=1)
            aug_input = torch.cat((input.half(), memory.half()), dim=1)
        else:
            # aug_input = torch.cat((input.float(), aux.float(), memory), dim=1)
            aug_input = torch.cat((input.float(), memory), dim=1)
        features = self.pre_transform.forward(aug_input)
        features = self.layers.forward(features.view(-1,self.transformer_length,self.transformer_input_size))
        action = self.output_0.forward(features.view(-1,self.matrix_size))
        discard = self.output_1.forward(features.view(-1,self.matrix_size))
        noble = self.output_2.forward(features.view(-1,self.matrix_size))
        memory = torch.tanh(self.output_memory.forward(features.view(-1,self.matrix_size)))
        return action, discard, noble, memory