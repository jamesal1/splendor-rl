import torch
from torch import nn
import torch.nn.functional as F
from constants import *
cuda_device = torch.device("cuda")
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


class Model(nn.Module):


    def __init__(self, **kwargs):
        super(Model,self).__init__()
        self.half = kwargs.get("half",True)
        self.memory_size=kwargs.get("memory_size",512)
        pass

    def get_action(self, input, memory):
        if self.half:
            input = input.half()
        chip_sum = input[:,VIEW_PLAYER+CHIPS:VIEW_PLAYER+CHIPS+5] + input[:,VIEW_OTHER+CHIPS:VIEW_OTHER+CHIPS+5]
        reserve_mask = input[:, VIEW_PLAYER+RESERVED:VIEW_PLAYER+RESERVED+90]
        chip_mask = input[:, VIEW_PLAYER+CHIPS:VIEW_PLAYER+CHIPS+6] > 0

        purchase_mask = (input[:, VIEW_PLAY:VIEW_PLAY+90] + reserve_mask) * resource_mask(input)
        take_mask = (((chip_sum.view(-1,1,5) + take_cuda.view(1,PASS,5)) < 5).sum(axis=2) == 5).half()
        aux = torch.cat((purchase_mask,take_mask),dim=1)
        action, discard, noble, memory = self.forward(input, aux, memory)

        action = torch.exp(action - action.max())
        discard = torch.exp(discard - discard.max())
        noble = torch.exp(noble - noble.max())
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
        m.bias.data.fill_(0.01)




class DenseNet(Model):


    def __init__(self, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        layer_sizes = kwargs.get("layer_sizes", [1024] * 3)
        layers = []
        s = input_size + self.memory_size + aux_size
        for t in layer_sizes:
            layers += [nn.Linear(s,t)]
            layers += [nn.ReLU(inplace=True)]
            s = t
        self.layers = nn.Sequential(*layers)
        self.output_0 = nn.Linear(t, action_size)
        self.output_1 = nn.Linear(t, discard_size)
        self.output_2 = nn.Linear(t, noble_size)
        self.output_memory = nn.Linear(t, self.memory_size)
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
        self.pre_transform = nn.Linear(s, self.transformer_input_size*self.transformer_length)
        layers += [nn.ReLU(inplace=True)]
        layers += [torch.nn.TransformerEncoderLayer(self.transformer_input_size, transformer_heads,
                                                    dim_feedforward=512, dropout=0, activation='relu')]


        self.layers = nn.Sequential(*layers)
        self.output = nn
        self.output_0 = nn.Linear(self.matrix_size, action_size)
        self.output_1 = nn.Linear(self.matrix_size, discard_size)
        self.output_2 = nn.Linear(self.matrix_size, noble_size)
        self.output_memory = nn.Linear(self.matrix_size, self.memory_size)
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
        features = self.pre_transform.forward(aug_input)
        features = self.layers.forward(features.view(-1,self.transformer_length,self.transformer_input_size))
        action = self.output_0.forward(features.view(-1,self.matrix_size))
        discard = self.output_1.forward(features.view(-1,self.matrix_size))
        noble = self.output_2.forward(features.view(-1,self.matrix_size))
        memory = torch.tanh(self.output_memory.forward(features.view(-1,self.matrix_size)))
        return action, discard, noble, memory