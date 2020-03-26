from tqdm import trange
import os
import model
import datetime
import shutil
import torch
import environment
import random
from constants import cuda_on
random.seed(2)
torch.manual_seed(2)

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, PPOTORCHPOLICY

if cuda_on:
    cuda_device = torch.device("cuda")
else:
    cuda_device = torch.device("cpu")

class Trainer():

    def __init__(self, my_model, **kwargs):
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "") \
            .replace(":", "") \
            .replace(" ", "_")
        self.log_dir = os.path.join(os.getcwd(), "log/" + date)
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        for f in ["trainer.py", "model.py"]:
            src = os.path.join(os.getcwd(), f)
            dst = os.path.join(self.log_dir, f)
            shutil.copyfile(src, dst)
        self.model = my_model
        self.noise_scale = kwargs.get("noise_scale", 1e-3)
        # self.noise_scale_decay = 1 - kwargs.get("noise_scale_decay", 1e-3)
        self.noise_scale_decay = 1 - kwargs.get("noise_scale_decay", 0)
        self.lr = kwargs.get("lr", 1e-3)
        # self.lr_decay = kwargs.get("lr_decay", 1e-2)
        self.ave_delta_rate = kwargs.get("ave_delta_rate", .999)
        self.epochs = kwargs.get("epochs", 1000)
        self.batches_per_epoch = kwargs.get("batches_per_epoch",100)
        self.batch_size = kwargs.get("batch_size",1)
        self.half = kwargs.get("half", 0)

    def train(self):
        env = environment.SplendorEnv(self.batch_size)

        perturbed_model = type(self.model)()
        noise_model = type(self.model)()
        if cuda_on:
            perturbed_model = perturbed_model.cuda()
            noise_model = noise_model.cuda()
        if self.half:
            self.model = self.model.half()
            perturbed_model = perturbed_model.half()
            noise_model = noise_model.half()
        ave_delta = .1
        # opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay = 1e-3, eps=1e-3)
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0)



        for epoch in range(self.epochs):
            print("Epoch:",epoch)
            total_reward = 0.0
            total_game_length = 0.0
            total_cards = 0.0
            total_points = 0.0
            for _ in trange(self.batches_per_epoch):
                perturbed_model.load_state_dict(self.model.state_dict())
                perturbed_player = random.randrange(2)
                player_1, player_2 = (self.model, perturbed_model) if perturbed_player else (perturbed_model, self.model)
                init_score = 6 * random.randrange(2)
                top = random.randrange(2)

                with torch.no_grad():
                    #based on https://github.com/kayuksel/pytorch-ars/blob/master/ars_multiprocess.py
                    for p_param, n_param in zip(perturbed_model.parameters(), noise_model.parameters()):
                        n_param.data.normal_(std=self.noise_scale)
                        p_param.add_(n_param.data)

                    add_result, add_turns, add_cards, add_points = environment.run(player_1, player_2, size=self.batch_size, init_score=init_score, top=top)

                    add_score = add_result[perturbed_player].sum()/self.batch_size
                    for p_param, n_param in zip(perturbed_model.parameters(), noise_model.parameters()):
                        p_param.sub_(2 * n_param.data)

                    sub_result, sub_turns, sub_cards, sub_points = environment.run(player_1, player_2, size=self.batch_size, init_score=init_score, top=top)
                    sub_score = sub_result[perturbed_player].sum()/self.batch_size
                    total_reward += add_score + sub_score

                    total_game_length += add_turns.sum() + sub_turns.sum()
                    total_cards += add_cards + sub_cards
                    total_points += add_points + sub_points - init_score * 4 * self.batch_size

                    reward_delta = sub_score - add_score
                    step_size = reward_delta / (ave_delta + 1e-5)
                    ave_delta = self.ave_delta_rate * ave_delta + (1 - self.ave_delta_rate) * abs(reward_delta)

                for param, n_param in zip(self.model.parameters(), noise_model.parameters()):
                    param.grad = ((step_size / self.noise_scale) * n_param.data)
                    # print((param.grad**2).mean())
                # exit()
                self.noise_scale *= self.noise_scale_decay
                opt.step()
            # for param in self.model.parameters():
            #     print ((param.data**2).mean())
            print("Average Reward:", total_reward / (2 * self.batches_per_epoch))
            print("Average Game Length:", total_game_length.float() / (2 * self.batches_per_epoch * self.batch_size))
            print("Average Cards:", total_cards.float() / (2 * self.batches_per_epoch * self.batch_size))
            print("Average Points:", total_points.float() / (2 * self.batches_per_epoch * self.batch_size))
            fname = os.path.join(self.checkpoints_dir, "epoch_"+str(epoch)+".pkl")
            torch.save(self.model, fname)


    def train_ppo(self):
        pass

def no_grad_test():
    import time
    for i in trange(100):
        time.sleep(1)

    var = torch.eye(5)
    var.grad = torch.ones((5, 5))

    print(var)



if __name__ == "__main__":
    my_model = model.DenseNet()
    # Trainer(model.TransformerNet()).train()
    if cuda_on:
        my_model = my_model.cuda()
    Trainer(my_model).train()