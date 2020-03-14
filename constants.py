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
