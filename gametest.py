import torch
from torch.utils.cpp_extension import load
game_cpp = load(name="game_cpp", sources=["game.cpp"])
# import game_cpp
from constants import *


decks = game_cpp.init_decks(10,0)
states = torch.zeros([10,492],dtype=torch.int8)
game_cpp.init_states(states, decks, 6, 0)
game_cpp.advance(states,decks,torch.randint(0,PURCHASE_END,(10,)).byte(),torch.randint(0,6,(10,3)).byte(),torch.randint(0,10,(10,)).byte())
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