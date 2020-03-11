import torch
from torch.utils.cpp_extension import load
# import os
# os.environ["CC"] = "/usr/bin/gcc-7"
# os.environ["CXX"] = "/usr/bin/g++-7"
game_cpp = load(name="game_cpp", sources=["game.cpp"])
# import game_cpp

decks = game_cpp.init_decks(10,0)
print(decks)
states = torch.zeros([10,492],dtype=torch.int8)
game_cpp.init_states(states, decks, 6, 0)
print(states[0])
print(states[1])
print(states)
# game_cpp.comp()
# t=game_cpp.get_test_vector()
# print(t)
# game_cpp.test_vector(t)

#game_cpp.advance(b,b)
# print(a)
# print(b)
#
# import time
# time.sleep(100)