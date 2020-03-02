import torch
from torch.utils.cpp_extension import load
# import os
# os.environ["CC"] = "/usr/bin/gcc-7"
# os.environ["CXX"] = "/usr/bin/g++-7"
game_cpp = load(name="game_cpp", sources=["game.cpp"])
# import game_cpp

a = torch.zeros([2,492],dtype=torch.int8).cuda()

b=a.cpu()
game_cpp.advance(b,b)
print(a)
print(b)

import time
time.sleep(100)