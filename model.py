import torch
from torch import nn
import torch.nn.functional as F
from constants import *
cuda_device = torch.device("cuda")


input_size = 491
action_size = 214
discard_size = 18
noble_size = 10

class Model(nn.Module):


    def __init__(self, **kwargs):
        super(Model,self).__init__()

        self.memory_size=kwargs.get("memory_size",512)
        pass

    def get_action(self, input, memory, turn):
        action, discard, noble, memory = self.forward(input, memory)
        action = torch.exp(action - action.max())
        discard = torch.exp(discard - discard.max())
        noble = torch.exp(noble - noble.max())
        reserve_mask = input[:, PLAYER_2+RESERVED:PLAYER_2+RESERVED+90] if turn else input[:, PLAYER_1+RESERVED:PLAYER_1+RESERVED+90]
        chip_mask = input[:, PLAYER_2+CHIPS:PLAYER_2+CHIPS+6] if turn else input[:, PLAYER_1+CHIPS:PLAYER_1+CHIPS+6]
        action[:, PURCHASE_START:PURCHASE_END] *= input[:, PLAY:PLAY+90] + reserve_mask
        action[:, RESERVE_START:RESERVE_END] *= input[:, PLAY:PLAY+90]
        _, action = torch.max(action, dim=1)

        _, discard = torch.max(discard.view(-1, 3, 6) * (chip_mask > 0).view(-1, 1, 6), dim=2)
        noble *= input[:,NOBLES:NOBLES + 10]
        _, noble = torch.max(noble, dim=1)

        return action, discard, noble, memory


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)




class DenseNet(Model):


    def __init__(self, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        layer_sizes = kwargs.get("layer_sizes", [2048] * 2)
        layers = []
        s = input_size + self.memory_size
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




    def forward(self, input, memory):
        if memory is None:
            memory = torch.zeros(input.size(0), self.memory_size, device=cuda_device)
        if self.output_0.weight.dtype == torch.float16:
            aug_input = torch.cat((input.half(), memory.half()), dim=1)
        else:
            aug_input = torch.cat((input.float(), memory), dim=1)

        features = self.layers.forward(aug_input)
        action = self.output_0.forward(features)
        discard = self.output_1.forward(features)
        noble = self.output_2.forward(features)
        memory = torch.tanh(self.output_memory.forward(features))
        return action, discard, noble, memory


