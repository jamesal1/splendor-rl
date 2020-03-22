#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <experimental/iterator>
#include <random>

typedef std::vector<std::vector<int>> Array2d;

//Level,Wcost,Ucost,Gcost,Rcost,Acost,Points,W,U,G,R,A
const int LEVEL = 0;
const int COST = 1;
const int POINTS = 6;
const int COLOR = 7;

const int PLAYER_1 = 0;
const int PLAYER_2 = 105;
const int SCORE = 0;
const int CHIPS = 1;
const int GOLD = 6;
const int CARDS = 7;
const int L1_RESERVED = 12;
const int L2_RESERVED = 13;
const int L3_RESERVED = 14;
const int RESERVED = 15;
const int PLAY = 210;
const int DEAD = 300;
const int NOBLES = 390;
const int TURN = 400;
const int TOP = 401;
const int RESULT = 491;

const int FIRST_CARD = 16;
const int MAX_TURNS = 90;

signed char CARD_INFO [90][12]= {
{1,0,3,0,0,0,0,1,0,0,0,0},
{1,0,0,0,2,1,0,1,0,0,0,0},
{1,0,1,1,1,1,0,1,0,0,0,0},
{1,0,2,0,0,2,0,1,0,0,0,0},
{1,0,0,4,0,0,1,1,0,0,0,0},
{1,0,1,2,1,1,0,1,0,0,0,0},
{1,0,2,2,0,1,0,1,0,0,0,0},
{1,3,1,0,0,1,0,1,0,0,0,0},
{1,1,0,0,0,2,0,0,1,0,0,0},
{1,0,0,0,0,3,0,0,1,0,0,0},
{1,1,0,1,1,1,0,0,1,0,0,0},
{1,0,0,2,0,2,0,0,1,0,0,0},
{1,0,0,0,4,0,1,0,1,0,0,0},
{1,1,0,1,2,1,0,0,1,0,0,0},
{1,1,0,2,2,0,0,0,1,0,0,0},
{1,0,1,3,1,0,0,0,1,0,0,0},
{1,2,1,0,0,0,0,0,0,1,0,0},
{1,0,0,0,3,0,0,0,0,1,0,0},
{1,1,1,0,1,1,0,0,0,1,0,0},
{1,0,2,0,2,0,0,0,0,1,0,0},
{1,0,0,0,0,4,1,0,0,1,0,0},
{1,1,1,0,1,2,0,0,0,1,0,0},
{1,0,1,0,2,2,0,0,0,1,0,0},
{1,1,3,1,0,0,0,0,0,1,0,0},
{1,0,2,1,0,0,0,0,0,0,1,0},
{1,3,0,0,0,0,0,0,0,0,1,0},
{1,1,1,1,0,1,0,0,0,0,1,0},
{1,2,0,0,2,0,0,0,0,0,1,0},
{1,4,0,0,0,0,1,0,0,0,1,0},
{1,2,1,1,0,1,0,0,0,0,1,0},
{1,2,0,1,0,2,0,0,0,0,1,0},
{1,1,0,0,1,3,0,0,0,0,1,0},
{1,0,0,2,1,0,0,0,0,0,0,1},
{1,0,0,3,0,0,0,0,0,0,0,1},
{1,1,1,1,1,0,0,0,0,0,0,1},
{1,2,0,2,0,0,0,0,0,0,0,1},
{1,0,4,0,0,0,1,0,0,0,0,1},
{1,1,2,1,1,0,0,0,0,0,0,1},
{1,2,2,0,1,0,0,0,0,0,0,1},
{1,0,0,1,3,1,0,0,0,0,0,1},
{2,0,0,0,5,0,2,1,0,0,0,0},
{2,6,0,0,0,0,3,1,0,0,0,0},
{2,0,0,3,2,2,1,1,0,0,0,0},
{2,0,0,1,4,2,2,1,0,0,0,0},
{2,2,3,0,3,0,1,1,0,0,0,0},
{2,0,0,0,5,3,2,1,0,0,0,0},
{2,0,5,0,0,0,2,0,1,0,0,0},
{2,0,6,0,0,0,3,0,1,0,0,0},
{2,0,2,2,3,0,1,0,1,0,0,0},
{2,2,0,0,1,4,2,0,1,0,0,0},
{2,0,2,3,0,3,1,0,1,0,0,0},
{2,5,3,0,0,0,2,0,1,0,0,0},
{2,0,0,5,0,0,2,0,0,1,0,0},
{2,0,0,6,0,0,3,0,0,1,0,0},
{2,2,3,0,0,2,1,0,0,1,0,0},
{2,3,0,2,3,0,1,0,0,1,0,0},
{2,4,2,0,0,1,2,0,0,1,0,0},
{2,0,5,3,0,0,2,0,0,1,0,0},
{2,0,0,0,0,5,2,0,0,0,1,0},
{2,0,0,0,6,0,3,0,0,0,1,0},
{2,2,0,0,2,3,1,0,0,0,1,0},
{2,1,4,2,0,0,2,0,0,0,1,0},
{2,0,3,0,2,3,1,0,0,0,1,0},
{2,3,0,0,0,5,2,0,0,0,1,0},
{2,5,0,0,0,0,2,0,0,0,0,1},
{2,0,0,0,0,6,3,0,0,0,0,1},
{2,3,2,2,0,0,1,0,0,0,0,1},
{2,0,1,4,2,0,2,0,0,0,0,1},
{2,3,0,3,0,2,1,0,0,0,0,1},
{2,0,0,5,3,0,2,0,0,0,0,1},
{3,0,0,0,0,7,4,1,0,0,0,0},
{3,3,0,0,0,7,5,1,0,0,0,0},
{3,3,0,0,3,6,4,1,0,0,0,0},
{3,0,3,3,5,3,3,1,0,0,0,0},
{3,7,0,0,0,0,4,0,1,0,0,0},
{3,7,3,0,0,0,5,0,1,0,0,0},
{3,6,3,0,0,3,4,0,1,0,0,0},
{3,3,0,3,3,5,3,0,1,0,0,0},
{3,0,7,0,0,0,4,0,0,1,0,0},
{3,0,7,3,0,0,5,0,0,1,0,0},
{3,3,6,3,0,0,4,0,0,1,0,0},
{3,5,3,0,3,3,3,0,0,1,0,0},
{3,0,0,7,0,0,4,0,0,0,1,0},
{3,0,0,7,3,0,5,0,0,0,1,0},
{3,0,3,6,3,0,4,0,0,0,1,0},
{3,3,5,3,0,3,3,0,0,0,1,0},
{3,0,0,0,7,0,4,0,0,0,0,1},
{3,0,0,0,7,3,5,0,0,0,0,1},
{3,0,0,3,6,3,4,0,0,0,0,1},
{3,3,3,5,3,0,3,0,0,0,0,1}};



signed char TAKE_INFO [26][5]= {
{1,1,1,0,0},
{1,1,0,1,0},
{1,1,0,0,1},
{1,0,1,1,0},
{1,0,1,0,1},
{1,0,0,1,1},
{0,1,1,1,0},
{0,1,1,0,1},
{0,1,0,1,1},
{0,0,1,1,1},
{1,1,0,0,0},
{1,0,1,0,0},
{1,0,0,1,0},
{1,0,0,0,1},
{0,1,1,0,0},
{0,1,0,1,0},
{0,1,0,0,1},
{0,0,1,1,0},
{0,0,1,0,1},
{0,0,0,1,1},
{1,0,0,0,0},
{0,1,0,0,0},
{0,0,1,0,0},
{0,0,0,1,0},
{0,0,0,0,1},
{0,0,0,0,0}};

const int L1_END = 40;
const int L2_END = 70;
const int L3_END = 90;

const int TAKE_START = 0;
const int TAKE_END = 26;
const int TAKE2_START = 26;
const int TAKE2_END = 31;
const int RESERVE_START = 31;
const int RESERVE_END = 121;
const int RESERVE_TOP_START = 121;
const int RESERVE_TOP_END = 124;
const int PURCHASE_START = 124;
const int PURCHASE_END = 214;

std::vector<int> DECK_1{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39};
std::vector<int> DECK_2{40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69};
std::vector<int> DECK_3{70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89};



signed char NOBLE_INFO [10][5]= {
{3,3,0,0,3},
{3,3,3,0,0},
{0,3,3,3,0},
{0,0,3,3,3},
{3,0,0,3,3},
{4,0,0,0,4},
{4,4,0,0,0},
{0,4,4,0,0},
{0,0,4,4,0},
{0,0,0,4,4}};

//struct vector_array {
//    std::vector<int> array;
//    int length;
//};


Array2d init_decks(int num, unsigned seed) {
    std::mt19937 rng(seed);
    Array2d decks;
    for (int i = 0; i < num; i++) {
        decks.push_back(DECK_1);
        std::shuffle (decks.back().begin(), decks.back().end(), rng);
        decks.push_back(DECK_2);
        std::shuffle (decks.back().begin(), decks.back().end(), rng);
        decks.push_back(DECK_3);
        std::shuffle (decks.back().begin(), decks.back().end(), rng);
    }

    return decks;
}

std::vector<int>* get_test_vector() {
    std::vector<int> *vects = new std::vector<int>[1];
    vects[0]=std::vector<int>{1,2};
    printf("okay\n");
    printf("%d\n",vects[0].back());
    return vects;
}

int test_vector (Array2d deck, int length) {
//    deck = init_decks(length);
    for (int i = 0; i < length;i++ ){
        printf("%d %d\n",i,deck.at(i).back());

    }
    return 0;
}

void init_states(torch::Tensor states, Array2d decks, int start_score, unsigned seed) {
    std::mt19937 rng(seed + 1);
    std::uniform_int_distribution<int> random10(0,9);
    std::uniform_int_distribution<int> random9(1,9);
    std::uniform_int_distribution<int> random8(2,9);
    auto statesd = states.accessor<signed char,2>();

    for (int i = 0; i< statesd.size(0); i++) {
        auto state = statesd[i];
        auto deck_1 = decks.at(3 * i);
        auto deck_2 = decks.at(3 * i + 1);
        auto deck_3 = decks.at(3 * i + 2);
        state[PLAY + deck_1.back()] = 1; deck_1.pop_back();
        state[PLAY + deck_1.back()] = 1; deck_1.pop_back();
        state[PLAY + deck_1.back()] = 1; deck_1.pop_back();
        state[PLAY + deck_1.back()] = 1; deck_1.pop_back();
        state[PLAY + deck_2.back()] = 1; deck_2.pop_back();
        state[PLAY + deck_2.back()] = 1; deck_2.pop_back();
        state[PLAY + deck_2.back()] = 1; deck_2.pop_back();
        state[PLAY + deck_2.back()] = 1; deck_2.pop_back();
        state[PLAY + deck_3.back()] = 1; deck_3.pop_back();
        state[PLAY + deck_3.back()] = 1; deck_3.pop_back();
        state[PLAY + deck_3.back()] = 1; deck_3.pop_back();
        state[PLAY + deck_3.back()] = 1; deck_3.pop_back();
        state[TOP + deck_1.back()] = 1;
        state[TOP + deck_2.back()] = 1;
        state[TOP + deck_3.back()] = 1;
        state[PLAYER_1 + SCORE] = start_score;
        state[PLAYER_2 + SCORE] = start_score;
        auto first_noble = random10(rng);
        auto second_noble = (first_noble + random9(rng)) % 10;
        auto third_noble = (first_noble + random8(rng)) % 10;
        if (third_noble == second_noble) {
            third_noble = (first_noble + 1) % 10;
        }
        state[NOBLES + first_noble] = 1;
        state[NOBLES + second_noble] = 1;
        state[NOBLES + third_noble] = 1;
    }
}



int advance(torch::Tensor states, Array2d decks,  torch::Tensor actions, torch::Tensor discards, torch::Tensor nobles) {
    auto statesd = states.accessor<signed char,2>();
    auto actionsd = actions.accessor<unsigned char,1>();
    auto discardsd = discards.accessor<unsigned char,2>();
    auto noblesd = nobles.accessor<unsigned char,1>();
    int games_unfinished = 0;
    for (int i = 0; i< statesd.size(0); i++) {
        auto state = statesd[i];
        if (state[RESULT]) {
            continue;
        }
        int player = (state[TURN] %  2);
        state[TURN]++;
        int player_offset = player * PLAYER_2 + (1 - player) * PLAYER_1;
        int replace_card = -1;
        int replace_top = -1;
        std::vector<int> deck;
        auto action = actionsd[i];
        if (action < TAKE_END) {
            auto chips = TAKE_INFO[action];
            for (int color = 0; color < 5; color++) {
                state[player_offset + CHIPS + color] += chips[color];
                if (state[PLAYER_1 + CHIPS + color] + state[PLAYER_2 + CHIPS + color] > 4) {
                    state[RESULT] = player ? 2:-2;
                }
            }
        } else if (action < TAKE2_END) {
            int color = action - TAKE2_START;
            if (state[PLAYER_1 + CHIPS + color] + state[PLAYER_2 + CHIPS + color]) {
                state[RESULT] = player ? 2:-2;
            } else {
                state[player_offset + CHIPS + color] += 2;
            }
        } else if (action < RESERVE_TOP_END) {
            int card;
            if (action < RESERVE_END) {
                card = action - RESERVE_START;
                if (state[PLAY + card]) {
                    replace_card = card;
                    state[DEAD + card] = 1;
                } else {
                    state[RESULT] = player ? 2:-2;
                }
            } else {
                int level = action - RESERVE_TOP_START;
                deck = decks.at(3 * i + level);
                if (deck.size()) {
                    card = deck.back();
                    replace_top = card;
                } else {
                    state[RESULT] = player ? 2:-2;
                }
            }
            if (state[player_offset + L1_RESERVED] + state[player_offset + L2_RESERVED] +
              state[player_offset + L3_RESERVED] < 3 ) {
                state[player_offset + RESERVED + card] = 1;
                if (card < L1_END) {
                    state[player_offset + L1_RESERVED]++;
                } else if (card < L2_END) {
                    state[player_offset + L2_RESERVED]++;
                } else {
                    state[player_offset + L3_RESERVED]++;
                }
                if (state[PLAYER_1 + GOLD] + state[PLAYER_2 + GOLD] < 5) {
                    state[player_offset + GOLD]++;
                }
            } else {
                state[RESULT] = player ? 2:-2;
            }
        } else {
            int card = action - PURCHASE_START;
            if (state[PLAY + card]) {
                replace_card = card;
            } else if (state[player_offset + RESERVED + card]) {
                int level = card < L1_END ? 0 : card < L2_END ? 1 : 2;
                state[player_offset + RESERVED + card] = 0;
                state[player_offset + L1_RESERVED + level]--;
            } else {
                state[RESULT] = player ? 2:-2;
            }
            int gold_cost = 0;
            auto card_info = CARD_INFO[card];
            for (int color = 0; color < 5; color++) {
                int cost = card_info[COST + color] - state[player_offset + CARDS + color];
                if (cost > 0) {
                    int chip = state[player_offset + CHIPS + color];
                    if (cost > chip) {
                        state[player_offset + CHIPS + color] = 0;
                        gold_cost += cost - chip;
                    } else {
                        state[player_offset + CHIPS + color] -= cost;
                    }
                }
                state[player_offset + CARDS + color] += card_info[COLOR + color];
            }
            state[player_offset + SCORE] += card_info[POINTS];
            state[player_offset + GOLD] -= gold_cost;
            state[DEAD + card] = 1;
            if (state[player_offset + GOLD] < 0) {
                state[RESULT] = player ? 2:-2;
            }
        }

        if (replace_card>-1) {
            int level = replace_card < L1_END ? 0 : replace_card < L2_END ? 1 : 2;
            deck = decks.at(3 * i + level);
            state[PLAY + replace_card] = 0;
            if (deck.size()) {
                state[PLAY + deck.back()] = 1;
                replace_top = deck.back();
            }

        }

        if (replace_top>-1) {
            state[TOP + replace_top] = 0;
            deck.pop_back();
            if (deck.size()) {
                state[TOP + deck.back()] = 1;
            }
        }

        int total_chips = state[player_offset + GOLD];
        for (int color = 0; color < 5; color++) {
            total_chips += state[player_offset + CHIPS + color];
        }
        int chip_count;
        auto discard = discardsd[i];
        switch (total_chips) {
            case 13:
                chip_count = state[player_offset + CHIPS + discard[2]];
                if (chip_count) {
                    state[player_offset + CHIPS + discard[2]] = chip_count - 1;
                } else {
                    state[RESULT] = player ? 2:-2;
                }
            case 12:
                chip_count = state[player_offset + CHIPS + discard[1]];
                if (chip_count) {
                    state[player_offset + CHIPS + discard[1]] = chip_count - 1;
                } else {
                    state[RESULT] = player ? 2:-2;
                }
            case 11:
                chip_count = state[player_offset + CHIPS + discard[0]];
                if (chip_count) {
                    state[player_offset + CHIPS + discard[0]] = chip_count - 1;
                } else {
                    state[RESULT] = player ? 2:-2;
                }
        }
        auto noble = noblesd[i];
        int satisfies=0;
        if (state[NOBLES + noble]) {
            satisfies = 1;
            for (int color = 0; color < 5; color++) {
                if (state[player_offset + CARDS + color] < NOBLE_INFO[noble][color]) {
                    satisfies = 0;
                    break;
                }
            }
        }
        if (satisfies) {
            state[player_offset + SCORE] += 3;
            state[NOBLES + noble] = 0;
        } else {
            for (int nob = 0; nob < 10; nob++) {
                if (state[NOBLES + nob]) {
                    satisfies = 1;
                    for (int color = 0; color < 5; color++) {
                        if (state[player_offset + CARDS + color] < NOBLE_INFO[nob][color]) {
                            satisfies = 0;
                            break;
                        }
                    }
                    if (satisfies) {
                        state[RESULT] = player ? 2:-2;
                        break;
                    }
                }

            }
        }

        if (player && state[RESULT] == 0) {
            auto p1_score = state[PLAYER_1 + SCORE];
            auto p2_score = state[PLAYER_2 + SCORE];

            int p1_cards = 0;
            int p2_cards = 0;
            for (int color = 0; color < 5; color++) {
             p1_cards += state[PLAYER_1 + CARDS + color];
             p2_cards += state[PLAYER_2 + CARDS + color];
            }
//            auto stalemate = state[TURN] > MAX_TURNS -  1 || (state[TURN] > FIRST_CARD && (p1_cards == 0 || p2_cards == 0));
            auto stalemate = state[TURN] > MAX_TURNS -  1 ;
            if (p1_score > 20 || p2_score > 20) {
                if (p1_score > p2_score) {
                    state[RESULT] = 1;
                } else if (p1_score < p2_score) {
                    state[RESULT] = -1;
                } else {
                    state[RESULT] = p1_cards > p2_cards ? -1 : p1_cards < p2_cards ? 1 : 3;
                }
            } else if (stalemate) {
                state[RESULT] = p1_cards > p2_cards ? 1 : p1_cards < p2_cards ? -1 : 4;
            }
        }


        if (state[RESULT]==0) {
            games_unfinished++;
        }

    }
    return games_unfinished;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("advance", &advance, "advance");
    m.def("init_decks", &init_decks, "init_decks");
    m.def("init_states", &init_states, "init_states");
    m.def("test_vector", &test_vector, "test_vector");
    m.def("get_test_vector", &get_test_vector, "get_test_vector");

}
