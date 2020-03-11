import time
from ws4py.client.threadedclient import WebSocketClient
import json
import string
import random
import pandas

card_table = pandas.read_csv("cards.csv").fillna(0,downcast="infer").values.tolist()
LEVEL = 0
COST = 1
POINTS = 6
COLOR = 7


PLAYER_1 = 0
PLAYER_2 = 105
SCORE = 0
CHIPS = 1
GOLD = 6
CARDS = 7
L1_RESERVED = 12
L2_RESERVED = 13
L3_RESERVED = 14
RESERVED = 15
PLAY = 210
DEAD = 300
NOBLES = 390
TURN = 400
TOP = 401
RESULT = 491

L1_END = 40
L2_END = 70
L3_END = 90

TAKE_START = 0
PASS = 25
TAKE_END = 26
TAKE2_START = 26
TAKE2_END = 31
RESERVE_START = 31
RESERVE_END = 121
RESERVE_TOP_START = 121
RESERVE_TOP_END = 124
PURCHASE_START = 124
PURCHASE_END = 214

TAKE_INFO = [
[1,1,1,0,0],
[1,1,0,1,0],
[1,1,0,0,1],
[1,0,1,1,0],
[1,0,1,0,1],
[1,0,0,1,1],
[0,1,1,1,0],
[0,1,1,0,1],
[0,1,0,1,1],
[0,0,1,1,1],
[1,1,0,0,0],
[1,0,1,0,0],
[1,0,0,1,0],
[1,0,0,0,1],
[0,1,1,0,0],
[0,1,0,1,0],
[0,1,0,0,1],
[0,0,1,1,0],
[0,0,1,0,1],
[0,0,0,1,1],
[1,0,0,0,0],
[0,1,0,0,0],
[0,0,1,0,0],
[0,0,0,1,0],
[0,0,0,0,1],
[0,0,0,0,0],
[2,0,0,0,0],
[0,2,0,0,0],
[0,0,2,0,0],
[0,0,0,2,0],
[0,0,0,0,2]]

class GamePlayer():

    def __init__(self, wsclient=None):
        self.client = wsclient

    def update_state(self,state):
        pass


class WSClient(WebSocketClient):

    def __init__(self,url, player=None, close_fun=None):
        self.alive = True
        self.url = url
        self.player = player
        self.player.client = self

        self.close_function = close_fun
        self.id_count = 1
        self.state = 0
        self.player_name = ""
        self.current_room = ""
        self.position = 0
        self.wins = 0
        self.ties = 0
        self.losses = 0
        super(WSClient,self).__init__(url)

    def opened(self):
        pass


    def closed(self, code, reason=None):
        self.alive = False
        print("Lost WS connection to",self.url, code, reason, time.time())
        if self.close_function is not None:
            self.close_function()

    def received_message(self, m):
        try:
            j = json.loads(json.loads(m.data[1:])[0])
            msg = j["msg"]
            if msg == "ping":
                self.send_dump({"msg":"pong"}, id_param=False)
            elif "collection" in j:
                col = j["collection"]
                if col =="rooms":
                    if msg == "changed":
                        fields = j["fields"]
                        if "spots" in fields:
                            spots = fields["spots"]
                            i=0
                            for spot in spots:
                                if "player" in spot:
                                    player = spot["player"]
                                    name = player["name"]
                                    if i==0 and name == self.player_name:
                                        print(name)
                                        self.current_room = j["id"]
                                        self.send_dump({"msg":"method","method":"proposeGame",
                                                "params":[{"roomId":self.current_room}], "randomSeed":getRandomLower()})
                        elif "state" in fields:
                            if fields["state"] == "Started":
                                self.send_dump({"msg":"sub", "id":getRandomId(), "name":"room",
                                                "params":[self.current_room]})
                elif col == "users":
                    if msg =="added":
                        self.player_name = j["fields"]["username"]
                        print("my name", self.player_name)
                elif col =="games":
                    fields = j["fields"]
                    if "status" in fields:
                        if fields["status"] == "FINISHING":
                            ranks = fields["result"]["ranks"]
                            if (ranks[0]>ranks[1])==self.position:
                                self.wins+=1
                            elif ranks[0]==ranks[1]:
                                self.ties+=1
                            else:
                                self.losses+=1
                            pass
                    if "activeStatuses" in fields:
                        job = fields["activeStatuses"][0]["job"]
                    actionItems = fields["actionItems"]
                    data = fields["data"]
                    print(data)
                    # if data:
                    #     print(list(fields))
                    # if actionItems:
                    #     print(actionItems[0]["action"])

                    # if "status" in fields:
                    #     if fields["status"] == "INPROGRESS":
                    #         print(fields)
                    #     else:
                    #         print(fields)
        except:
            print(m.data)
        # self.callback(m.data)

    def send_dump(self, msg, id_param=True):
        if id_param and "id" not in msg:
            msg["id"]=str(self.id_count)
            self.id_count +=1
        self.send(json.dumps([json.dumps(msg)]))

    def create_room(self, score="21", top=False):
        room_config = [{"gameKey":"spendee","numPlayers":2,"numAIPlayers":1,
                        "gameSettings":{"speed":"fast","targetScore":score,"nextCardVisible":top}}]
        self.send_dump({"msg":"method","method":"newRoom","params":room_config, "randomSeed":getRandomLower()})


    def guest_login(self):
        self.send_dump({"msg":"method","method":"login","params":[{"createGuest":"true"}]})


    def connect_to_spendee(self):
        self.connect()
        self.send_dump({"msg":"connect","version":"1","support":["1","pre2","pre1"]},id_param=False)
        self.send_dump({"msg":"sub","id":getRandomId(),"name":"lobbyRooms","params":["spendee"]})


def parseState(data):
    state = [0]*492
    if data["targetScore"] == 15:
        state[PLAYER_1 + SCORE] += 6
        state[PLAYER_2 + SCORE] += 6
    player_offset = PLAYER_1
    for player in data["players"]:
        for color,count in enumerate(player["chips"]):
            state[player_offset + CHIPS + color] = count
        state[player_offset + GOLD] = player["goldChips"]
        for card in player["purchasedCards"]:
            state[player_offset + SCORE] += card_table[card][POINTS]
            for color in range(5):
                state[player_offset + CARDS + color] += card_table[card][COLOR + color]
            state[DEAD + card] = 1
        for card in player["reservedCards"]:
            state[player_offset + RESERVED + card] = 1
            level = card_table[card][LEVEL]
            state[player_offset + L1_RESERVED - 1 + level] += 1
        state[player_offset + SCORE] += 3 * len(player["nobles"])
        player_offset = PLAYER_2
    state[TURN] = 2 * data["turnsCount"] - 2 + data["state"]["currentPlayerIndex"]
    bank = data["bank"]
    for noble in bank["nobles"]:
        state[NOBLES + noble] = 1
    for four in bank["showedCards"]:
        for card in four:
            state[PLAY + card] = 1
    decks = bank["hiddenCards"]
    for deck in decks:
        state[TOP+deck[0]] = 1
    return state,decks
    #top



def formatAction(state,action,gameId,isAI = False):
    action_dict = {"playerIndex":state[TURN] % 2}

    if action == PASS:
        action_dict["type"] = "passRegular"
    elif action < TAKE2_END:
        action_dict["type"] = "pickChips"
        action_dict["chips"] = TAKE_INFO[action]
    elif action < RESERVE_END:
        action_dict["type"] = "reserveShowedCard"
        action_dict["cardIndex"] = action - RESERVE_START
    elif action < RESERVE_TOP_END:
        action_dict["type"] = "reserveHiddenCard"
        action_dict["level"] = action - RESERVE_TOP_START
    else:
        card = action - PURCHASE_START
        if state[PLAY + card]:
            action_dict["type"] = "buyCard"
        else:
            action_dict["type"] = "buyReservedCard"
        action_dict["cardIndex"] = card
    return {"msg":"method",
     "method":"gameAIPlayerAction" if isAI else "gamePlayerAction",
     "randomSeed": getRandomLower(),
     "params":[{"gameId":gameId, "action":action_dict}]}


def getRandomId():
    return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(17)])

def getRandomLower():
    return ''.join([random.choice("abcdef" + string.digits) for _ in range(20)])

def getWebsocket():
    link = "wss://spendee.mattle.online/sockjs/950/yv0cpudr/websocket"
    client = WSClient(link, GamePlayer())
    client.connect_to_spendee()
    time.sleep(1)
    client.guest_login()
    time.sleep(3)
    client.create_room()
    return client


getWebsocket()
time.sleep(100)

