import torch
from torch.utils.cpp_extension import load
game_cpp = load(name="game_cpp", sources=["game.cpp"])
# import game_cpp
import copy
import time
import threading
from constants import *
num_games = 1024
torch.manual_seed(0)
cores = 2

a = [0]
print(game_cpp.vector(a))
print(a)



# advances = [game_cpp.advance_1,game_cpp.advance_2,game_cpp.advance_3,game_cpp.advance_4]
#
# def parallel_advance(states, decks, actions, discards, nobles):
#     total = [0]
#     def help(i):
#         start = num * i
#         end = num * (i+1)
#         changed = advances[i](copy.deepcopy(states[start:end]), decks[3*start:3*end],copy.deepcopy(actions[start:end]),copy.deepcopy(discards[start:end]),copy.deepcopy(nobles[start:end]))
#         if changed:
#             total[0]+=changed
#     num = states.size(0)//cores
#
#     threads = []
#     for i in range(cores):
#     #     help(i)
#         threads+=[threading.Thread(target=help,args=(i,))]
#     for t in threads:
#         t.start()
#         # t.join()
#     for t in threads:
#         t.join()
#     return total[0]


def cpp_parallel_advance(states, decks, actions, discards, nobles):
    start_time = time.time()
    num = states.size(0)//cores
    statelist = []
    decklist = []
    actionlist = []
    discardlist = []
    noblelist = []

    for i in range(cores):
        start = num * i
        end = num * (i+1)
        statelist+=[states[start:end]]
        decklist+=[decks[3*start:3*end]]
        actionlist+=[actions[start:end]]
        discardlist+=[discards[start:end]]
        noblelist+=[nobles[start:end]]
    print("list", time.time() - start_time)
    return game_cpp.parallel_advance(statelist,decklist,actionlist,discardlist,noblelist)




for i in range(10):
    decks = torch.zeros([num_games,93]).byte()
    game_cpp.init_decks(decks,0)
    states = torch.zeros([num_games,492],dtype=torch.int8)
    game_cpp.init_states(states, decks, 6, 0)
    torch.manual_seed(0)
    start = time.time()
    print(game_cpp.advance(states,decks,torch.randint(0,PURCHASE_END,(num_games,)).byte(),torch.randint(0,6,(num_games,3)).byte(),torch.randint(0,10,(num_games,)).byte()))
    print("single",time.time()-start)
for i in range(10):
    decks2 = game_cpp.init_decks(num_games,0)
    states2 = torch.zeros([num_games,492],dtype=torch.int8)
    game_cpp.init_states(states2, decks2, 6, 0)
    torch.manual_seed(0)
    start = time.time()
    rand_actions= torch.randint(0,PURCHASE_END,(num_games,)).byte()
    rand_discards = torch.randint(0,6,(num_games,3)).byte()
    rand_nobles = torch.randint(0,10,(num_games,)).byte()
    print("rand",time.time()-start)
    start = time.time()

    # print(game_cpp.advance(states2,decks2,rand_actions,rand_discards,rand_nobles))
    # print(cpp_parallel_advance(states2,decks2,rand_actions,rand_discards,rand_nobles))
    print(game_cpp.nop([states2],[decks2]))
    print("threaded",time.time()-start)

def all_eq(a,b):
    return torch.all(torch.eq(a,b))
#
# print(torch.all(torch.eq(states,states2)))
#
# for i in range(states.size(0)):
#     if not all_eq(states[i],states2[i]):
#         for j in range(states.size(1)):
#             if states[i,j]!=states2[i,j]:
#                 print(i,j,states[i,j],states2[i,j])
# print(states[999])
# parallel_advance(states,decks,torch.randint(0,PURCHASE_END,(num_games,)).byte(),torch.randint(0,6,(num_games,3)).byte(),torch.randint(0,num_games,(num_games,)).byte())
# game_cpp.parallel_advance(states,decks,torch.randint(0,PURCHASE_END,(num_games,)).byte(),torch.randint(0,6,(num_games,3)).byte(),torch.randint(0,num_games,(num_games,)).byte())
# print(states[0])
# print(states[1])
# print(states)
# print(states.sum())
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