import torch
from torch.utils.cpp_extension import load
import os
os.environ["CC"] = "/usr/bin/gcc-7"
os.environ["CXX"] = "/usr/bin/g++-7"


lltm_cpp = load(name="game_cpp", sources=["game.cpp"])

print(torch.Tensor.eye(2))
torch.zeros([2,2],dtype=torch.uint8)