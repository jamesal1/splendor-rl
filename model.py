import torch
from torch import nn
import torch.nn.functional as F

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




def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)




class DenseNet(Model):


    def __init__(self, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        layer_sizes = kwargs.get("layer_sizes", [4096] * 3)
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
        aug_input = torch.cat((input.float(), memory), dim=1)
        features = self.layers.forward(aug_input)
        _, action = torch.max(self.output_0.forward(features), dim=1)
        _, discard = torch.max(self.output_1.forward(features).view(-1, 3, 6), dim=2)
        _, noble = torch.max(self.output_2.forward(features), dim=1)
        memory = torch.tanh(self.output_memory.forward(features))
        return action, discard, noble, memory


