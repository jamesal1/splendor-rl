from constants import *
import torch
from torch.utils.cpp_extension import load
game_cpp = load(name="game_cpp", sources=["game.cpp"])
import random
class Player(object):

    def __init__(self, wsclient=None):
        self.client = wsclient


    def get_move(self,job,state,decks):
        raise NotImplemented



def format_deck(decks):
    l1 = len(decks[0])
    l2 = len(decks[1])
    l3 = len(decks[2])
    return [0] * (40 - l1) + decks[0] + [0] * (30 - l2) + decks[1] + [0] * (20 - l3) + decks[2] +\
           [40 - l1 % 40, 70 - l2 % 70, 90 - l3 % 90]


class RandomPlayer(Player):


    def __init__(self,wsclient=None):
        super(RandomPlayer,self).__init__(wsclient)
        self.current_state = None

    def clear_state(self):
        self.current_state = None


    def get_move(self,job,state,deck):
        print("move")
        # return [PASS,0,0]
        if self.current_state != None and job == "SPENDEE_REGULAR":
            if self.expected_state !=state:
                print("State mismatch:")
                # print(self.expected_state)
                # print(state)
                for i in range(len(state)):
                    if state[i]!=self.expected_state[i]:
                        print(i,state[i],self.expected_state[i])
                # exit()



        if job == "SPENDEE_REGULAR":
            self.current_state = state
            player = state[TURN] % 2
            player_offset = player * PLAYER_2

            # chips = sum(state[player_offset+CHIPS:player_offset+GOLD+1])
            #
            # purchase_length = (PURCHASE_END-PURCHASE_START)
            # purchase_states = torch.tensor([state]*purchase_length)
            # purchase_decks = [decks.copy()] * purchase_length
            # purchase_actions = torch.tensor(range(PURCHASE_START,PURCHASE_END))
            # purchase_discards = torch.tensor([[0]*6]*purchase_length)
            # purchase_nobles = torch.randint(0,9,(purchase_length,))
            chips = sum(state[player_offset+CHIPS:player_offset+GOLD+1])
            states = torch.tensor([state]*PURCHASE_END).char()
            decks = torch.tensor([format_deck(deck)]*PURCHASE_END).byte()
            actions = torch.tensor(range(PURCHASE_END)).byte()
            discards = torch.randint(0,6,(PURCHASE_END,3)).byte()
            nobles = torch.randint(0,10,(PURCHASE_END,)).byte()
            # game_cpp.advance(purchase_states,purchase_decks,purchase_a)
            game_cpp.advance(states,decks,actions,discards,nobles)
            print(states[:,RESULT].tolist())
            legal = [i for i,x in enumerate(states[:,RESULT].tolist()) if abs(x)!=2]
            if len(legal)>1:
                legal.remove(PASS)
            chosen = random.choice(legal)
            self.expected_state = states[chosen].tolist()
            print(self.expected_state)
            return [chosen,discards[chosen].tolist(),nobles[chosen].item()]
        return None

if __name__ == "__main__":
    player = RandomPlayer()

    # player.get_move("SPENDEE_REGULAR",)