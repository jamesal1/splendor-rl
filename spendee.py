import time
from ws4py.client.threadedclient import WebSocketClient
import json
import string
import random
from constants import *
import player
import traceback
from collections import OrderedDict

class WSClient(WebSocketClient):

    def __init__(self,url, player=None, close_fun=None):
        self.alive = True
        self.url = url
        self.player = player
        self.player.client = self
        self.saved_command = None
        self.close_function = close_fun
        self.id_count = 1
        self.state = 0
        self.started = False
        self.player_name = ""
        self.current_room = ""
        self.game_id = ""
        self.position = 0
        self.vs_ai = False
        self.move = None
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
                                if not self.started:
                                    self.game_id = fields["gameId"]
                                    self.send_dump({"msg":"sub", "id":getRandomId(), "name":"room",
                                                    "params":[self.current_room]})
                                    self.send_dump({"msg":"sub", "id":getRandomId(), "name":"roomActiveUsers",
                                                    "params":[self.current_room]})
                                    self.send_dump({"msg":"method","method":"userPing",
                                            "params":[{"gameKey":"spendee","location":"/room/"+self.current_room,"isFocus":True}]})
                                    self.started = True
                elif col == "users":
                    if msg == "added":
                        self.player_name = j["fields"]["username"]
                        print("my name", self.player_name)
                elif col =="games":
                    fields = j["fields"]

                    if "status" in fields:
                        if msg == "added":
                            i = 0
                            for player in fields["players"]:
                                if player["isHuman"]:
                                    if player["name"] == self.player_name:
                                        self.position = i
                                else:
                                    self.vs_ai = True
                                i += 1
                        elif fields["status"] == "FINISHING":
                            ranks = fields["result"]["ranks"]
                            if (ranks[0]>ranks[1])==self.position:
                                self.wins += 1
                            elif ranks[0]== ranks[1]:
                                self.ties += 1
                            else:
                                self.losses += 1

                    elif "activeStatuses" in fields:
                        print("okay")
                        job = fields["activeStatuses"][0]["job"]
                        data = fields["data"]
                        if self.position == data["state"]["currentPlayerIndex"]:
                            ai_move = False
                        elif self.vs_ai:
                            ai_move = True
                        else:
                            return
                        if job == "SPENDEE_FLIP_NEW_CARD":
                            return
                        state,decks = parseState(data)
                        move = self.player.get_move(job,state,decks)
                        if move is not None:
                            self.move = move
                        if job == "SPENDEE_REGULAR":
                            print(data)
                            # {"msg":"method","method":"gamePlayerAction","params":[{"gameId":"eAu5PA9FDN5ytgajc","action":{"type":"passRegular","playerIndex":0}}],"id":"4","randomSeed":"0f698f24c3ab91836f74"}

                            self.save_command(formatRegular(state, self.move[0], self.game_id, ai_move), random_seed=True)
                            self.send_saved_command()
                        elif job == "SPENDEE_RETURN_CHIPS":
                            self.save_command(formatReturn(state, self.move[1], self.game_id, ai_move), random_seed=True)
                            self.send_saved_command()
                        elif job == "SPENDEE_PICK_NOBLE":
                            self.save_command(formatNoble(state, self.move[2], self.game_id, ai_move), random_seed=True)
                            self.send_saved_command()
            elif msg == "result" and "error" in j:
                print(j)
                self.send_saved_command()
            # print(j)
        except Exception as e:
            traceback.print_exc()
            print(e)
            print(m.data)
        # self.callback(m.data)

    def send_saved_command(self):
        if self.saved_command:
            self.send_dump(*self.saved_command)

    def save_command(self, msg, id_param=True,random_seed=False):
        self.saved_command = (msg,id_param,random_seed)

    def send_dump(self, msg, id_param=True,random_seed=False):
        msg = msg.copy()
        if id_param and "id" not in msg:
            msg["id"]=str(self.id_count)
            self.id_count +=1
        if random_seed:
            msg["randomSeed"] = getRandomLower()
        payload = json.dumps([json.dumps(msg)]).replace(" ","")
        print("Sent:", json.dumps(msg))
        self.send(payload)

    def create_room(self, score="21", top=True):
        room_config = [{"gameKey":"spendee","numPlayers":2,"numAIPlayers":1,
                        "gameSettings":{"speed":"fast","targetScore":score,"nextCardVisible":top}}]
        self.send_dump({"msg":"method","method":"newRoom","params":room_config, "randomSeed":getRandomLower()})


    def guest_login(self):
        self.send_dump({"msg":"method","method":"login","params":[{"createGuest":"true"}]})


    def connect_to_spendee(self):
        self.connect()
        self.send_dump({"msg":"connect","version":"1","support":["1","pre2","pre1"]},id_param=False)
        self.send_dump({"msg":"sub","id":getRandomId(),"name":"lobbyRooms","params":["spendee"]})
        self.send_dump({"msg":"sub","id":getRandomId(),"name":"meteor.loginServiceConfiguration","params":[]})
        self.send_dump({"msg":"sub","id":getRandomId(),"name":"_roles","params":[]})
        self.send_dump({"msg":"sub","id":getRandomId(),"name":"meteor_autoupdate_clientVersions","params":[]})


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
            state[player_offset + SCORE] += CARD_INFO[card][POINTS]
            for color in range(5):
                state[player_offset + CARDS + color] += CARD_INFO[card][COLOR + color]
            state[DEAD + card] = 1
        for card in player["reservedCards"]:
            state[player_offset + RESERVED + card] = 1
            level = CARD_INFO[card][LEVEL]
            state[player_offset + L1_RESERVED - 1 + level] += 1
        state[player_offset + SCORE] += 3 * len(player["nobles"])
        player_offset = PLAYER_2
    state[TURN] = 2 * data["turnsCount"] - 2 + data["state"]["currentPlayerIndex"]
    bank = data["bank"]
    for noble in bank["nobles"]:
        state[NOBLES + noble] = 1
    print(bank["showedCards"])
    for four in bank["showedCards"]:
        for card in four:
            if card is not None:
                state[PLAY + card] = 1
    decks = bank["hiddenCards"]
    for deck in decks:
        if deck:
            state[TOP+deck[0]] = 1
    return state,decks


def formatAction(state,action_dict,gameId,isAI=False):
    if "playerIndex" not in action_dict:
        action_dict["playerIndex"] = state[TURN] % 2

    params = OrderedDict([("gameId",gameId)])
    if isAI:
        params["playerIndex"] = action_dict["playerIndex"]
    params["action"] = action_dict
    ret = OrderedDict([("msg","method"),
                       ("method","gameAIPlayerAction" if isAI else "gamePlayerAction"),
                       ("params",[params])])
    print(json.dumps(ret))
    return ret


def formatRegular(state,action,gameId,isAI = False):
    action_dict = OrderedDict()
    if action == PASS:
        action_dict["type"] = "passRegular"
    elif action < TAKE2_END:
        action_dict["type"] = "pickChips"
        action_dict["playerIndex"] = state[TURN] % 2
        action_dict["chips"] = TAKE_INFO[action]
    elif action < RESERVE_END:
        action_dict["type"] = "reserveShowedCard"
        action_dict["playerIndex"] = state[TURN] % 2
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
        action_dict["playerIndex"] = state[TURN] % 2
        action_dict["cardIndex"] = card

    return formatAction(state,action_dict,gameId,isAI)

def formatReturn(state,action,gameId,isAI = False):
    turn = (state[TURN] % 2)
    player_offset = turn * PLAYER_2 + (1-turn) * PLAYER_1
    excess = sum(state[player_offset+CHIPS:player_offset+GOLD+1]) - 10
    chips = [0] * 6
    for i in range(excess):
        chips[action[i]] += 1
    action_dict = OrderedDict([("type","returnChips"),("playerIndex",turn),("goldChips",chips[5]),("chips",chips[:5])])
    return formatAction(state,action_dict,gameId,isAI)

def formatNoble(state,action,gameId,isAI = False):
    action_dict = {"playerIndex":state[TURN] % 2,"type":"pickNoble","nobleIndex":action}
    return formatAction(state,action_dict,gameId,isAI)

def getRandomId():
    return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(17)])

def getRandomLower():
    return ''.join([random.choice("abcdef" + string.digits) for _ in range(20)])

def getWebsocket():
    link = "wss://spendee.mattle.online/sockjs/950/yv0cpudr/websocket"
    client = WSClient(link, player.RandomPlayer())
    client.connect_to_spendee()
    time.sleep(1)
    client.guest_login()
    time.sleep(3)
    client.create_room()
    return client


getWebsocket()
time.sleep(100)

