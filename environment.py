import torch
from torch.utils.cpp_extension import load
game_cpp = load(name="game_cpp", sources=["game.cpp"])
from constants import *
import random
import time



def get_view(states,turn,top=False):
    view = states[:, :RESULT].cuda()
    player = turn % 2
    other_player_offset = PLAYER_1 if player else PLAYER_2
    view[:,other_player_offset+RESERVED:other_player_offset+RESERVED+90]
    if not top:
        view[:, TOP:RESULT] = 0
    view[:, TURN] = player
    return view

def parse_results(results):
    ret1 = []
    ret2 = []
    for r in results:
        if r == 1:
            ret1 += [1.]
            ret2 += [0.]
        elif r == -1:
            ret1 += [0.]
            ret2 += [1.]
        elif r == 2:
            ret1 += [.5]
            ret2 += [0.]
        elif r == -2:
            ret1 += [0.]
            ret2 += [.5]
        elif r == 3:
            ret1 += [.5]
            ret2 += [.5]
        elif r == 4:
            ret1 += [0]
            ret2 += [0]
    return torch.tensor(ret1).cuda(), torch.tensor(ret2).cuda()


def run(model_1, model_2, size, init_score=6, top=False):
    decks = game_cpp.init_decks(size, random.randrange(1000000000))
    states = torch.zeros([size, 492], dtype=torch.int8)
    game_cpp.init_states(states, decks, init_score, random.randrange(1000000000))
    turn = 0
    changed = 1
    p1_memory = None
    p2_memory = None
    while changed:
        view = get_view(states, turn, top)
        start = time.time()
        actions, discards, nobles, p1_memory = model_1.get_action(view, p1_memory, 0)
        # print ("nntime:",time.time() - start)
        start = time.time()
        game_cpp.advance(states, decks, actions.byte().cpu(), discards.byte().cpu(), nobles.byte().cpu())
        # print("advtime:", time.time() - start)
        turn += 1
        view = get_view(states, turn, top)
        actions, discards, nobles, p2_memory = model_2.get_action(view, p2_memory, 1)
        changed = game_cpp.advance(states, decks, actions.byte().cpu(), discards.byte().cpu(), nobles.byte().cpu())
        turn += 1
        if turn>100:
            print("stuck")
    return parse_results(states[:, RESULT].tolist()), states[:, TURN].cuda(), states[:,CARDS:CARDS+5].sum() + states[:,PLAYER_2+CARDS:PLAYER_2+CARDS+5].sum()
