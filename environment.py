import torch
from torch.utils.cpp_extension import load
game_cpp = load(name="game_cpp", sources=["game.cpp"])
from constants import *
import random
import time
from threading import Thread
import gym, ray



#
# class SplendorEnv(ray.MultiAgentEnv):
#
#     def __init__(self,**kwargs):
#         super(SplendorEnv,self).__init__()
#         self.batch_size = kwargs.get("batch_size",1)
#         self.set_game_options()
#
#
#     def set_game_options(self,init_score = 6, top = False):
#         self.init_score = init_score
#         self.top = top
#
#
#
#     def reset(self):
#         self.turn = 0
#         self.decks = game_cpp.init_decks(self.batch_size, random.randrange(1000000000))
#         self.states = torch.zeros([self.batch_size, 492], dtype=torch.int8)
#         game_cpp.init_states(self.states, self.decks, self.init_score, random.randrange(1000000000))
#         return {"p1":get_view(self.states, self.turn, self.top)}
#
#
#     def step(self, action_dict):
#         player = self.turn % 2
#         action, discard, noble = action_dict["p2" if player else "p1"]
#         changed = game_cpp.advance(self.states, self.decks, action, discard, noble)
#         self.turn += 1
#         if changed:
#             obs = {"p1" if player else "p2": get_view(self.states,self.turn,self.top)}
#             return obs, {}, {"__all__":False}, {}
#         else:
#             res1,res2 = parse_results(self.states)
#             return {}, {"p1": res1, "p2":res2}, {"__all__":True}, {}



#
#
#
# class MyEnv(gym.Env):
#     def __init__(self, env_config):
#         self.action_space = <gym.Space>
#         self.observation_space = <gym.Space>
#     def reset(self):
#         return <obs>
#     def step(self, action):
#         return <obs>, <reward: float>, <done: bool>, <info: dict>


def parallel_advance(states, decks, actions, discards, nobles):
    total = [0]
    def help(i):
        start = num * i
        end = num * (i+1)
        changed = game_cpp.advance(states[start:end], decks[3*start:3*end], actions[start:end], discards[start:end], nobles[start:end])
        if changed:
            total[0]+=changed
    num = states.size(0)//cores
    threads = []
    for i in range(cores):
        threads+=[Thread(target=help,args=(i,))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return total[0]


def get_view(states,turn,top=False):
    if cuda_on:
        states = states.cuda()
    player = turn % 2
    player_offset, other_player_offset = (PLAYER_2,PLAYER_1) if player else (PLAYER_1, PLAYER_2)
    view = torch.cat((states[:, player_offset:player_offset+PLAYER_LENGTH],
                      states[:, other_player_offset:other_player_offset+OTHER_PLAYER_LENGTH],
                      states[:, PLAY:RESULT]), dim=1)
    if not top:
        view[:, VIEW_TOP:] = 0
    view[:, VIEW_TURN] = player
    return view

def parse_results(results):
    start = time.time()
    p1_cards = results[:, PLAYER_1 + CARDS: PLAYER_1 + CARDS + 5].sum(dim=1)
    p2_cards = results[:, PLAYER_2 + CARDS: PLAYER_2 + CARDS + 5].sum(dim=1)
    total_cards = p1_cards + p2_cards + 1.0
    r = results[:,RESULT]
    # print(r)
    ret2 = (r<0).float() + (r==3).float() * .5
    stalemate = (r==4).float()
    ret1 = 1 - ret2 - stalemate
    # ret1 += stalemate * p1_cards / total_cards
    # ret2 += stalemate * p2_cards / total_cards
    ret1 += stalemate * p1_cards > p2_cards
    ret2 += stalemate * p2_cards > p1_cards
    return ret1,ret2
    p1_gold = results[:,GOLD]
    p2_gold = results[:,PLAYER_2+GOLD]
    # return p1_gold, p2_gold
    length = -results[:,TURN]/90.0
    return length,length


def run(model_1, model_2, size, init_score=6, top=False):
    decks = torch.zeros([size, 93]).byte()
    game_cpp.init_decks(decks, random.randrange(1000000000))
    states = torch.zeros([size, 492], dtype=torch.int8)
    game_cpp.init_states(states, decks, init_score, random.randrange(1000000000))
    # decks = game_cpp.init_decks(size, 0)
    # states = torch.zeros([size, 492], dtype=torch.int8)
    # game_cpp.init_states(states, decks, init_score, 0)
    turn = 0
    changed = 1
    p1_memory = None
    p2_memory = None
    while changed:
        start = time.time()
        view = get_view(states, turn, top)
        # print ("viewtime:", time.time() - start)
        start = time.time()
        actions, discards, nobles, p1_memory = model_1.get_action(view, p1_memory)
        # print ("nntime:",time.time() - start)
        start = time.time()
        game_cpp.advance(states, decks, actions.byte().cpu(), discards.byte().cpu(), nobles.byte().cpu())
        # parallel_advance(states, decks, actions.byte().cpu(), discards.byte().cpu(), nobles.byte().cpu())
        # print("advtime:", time.time() - start)
        turn += 1
        view = get_view(states, turn, top)
        start = time.time()
        actions, discards, nobles, p2_memory = model_2.get_action(view, p2_memory)
        # print ("nntime:",time.time() - start)
        start = time.time()
        actions, discards, nobles = actions.byte().cpu(), discards.byte().cpu(), nobles.byte().cpu()
        # print ("cputime:",time.time() - start)
        start = time.time()
        changed = game_cpp.advance(states, decks, actions, discards, nobles)
        # changed = parallel_advance(states, decks, actions.byte().cpu(), discards.byte().cpu(), nobles.byte().cpu())
        # print("advtime:", time.time() - start)
        turn += 1
        if turn>100:
            print("stuck")
    return parse_results(states), \
        states[:,TURN], \
        states[:,CARDS:CARDS+5].sum() + states[:,PLAYER_2+CARDS:PLAYER_2+CARDS+5].sum(), \
        states[:,SCORE].sum() + states[:,PLAYER_2+SCORE].sum()
