import torch
from torch.utils.cpp_extension import load
game_cpp = load(name="game_cpp", sources=["game.cpp"])
# import game_cpp
from constants import *
num_games = 8;
torch.manual_seed(0);
decks = game_cpp.init_decks(num_games,0)
states = torch.zeros([num_games,492],dtype=torch.int8)
game_cpp.init_states(states, decks, 6, 0)
game_cpp.advance(states,decks,torch.randint(0,PURCHASE_END,(num_games,)).byte(),torch.randint(0,6,(num_games,3)).byte(),torch.randint(0,num_games,(num_games,)).byte())
# game_cpp.parallel_advance(states,decks,torch.randint(0,PURCHASE_END,(num_games,)).byte(),torch.randint(0,6,(num_games,3)).byte(),torch.randint(0,num_games,(num_games,)).byte())
print(states[0])
print(states[1])
print(states)
print(states.sum())
# game_cpp.comp()
# t=game_cpp.get_test_vector()
# print(t)
# game_cpp.test_vector(t)

#game_cpp.advance(b,b)
# print(a)
# print(b)
#
# import time
# time.sleep(num_games0)