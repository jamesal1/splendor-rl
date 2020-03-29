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
import time

if cuda_on:
    cuda_device = torch.device("cuda")
else:
    cuda_device = torch.device("cpu")

class Trainer():

    def __init__(self, my_model, **kwargs):
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                     .replace(":", "")\
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
        self.noise_scale = kwargs.get("noise_scale", 3e-3)
        # self.noise_scale_decay = 1 - kwargs.get("noise_scale_decay", 1e-3)
        self.noise_scale_decay = 1 - kwargs.get("noise_scale_decay", 0)
        self.lr = kwargs.get("lr", 1e-3)
        # self.lr_decay = kwargs.get("lr_decay", 1e-2)
        self.ave_delta_rate = kwargs.get("ave_delta_rate", .99)
        self.epochs = kwargs.get("epochs", 1000)
        self.batches_per_epoch = kwargs.get("batches_per_epoch",100)
        self.batch_size = kwargs.get("batch_size",1)
        self.directions = kwargs.get("directions",1)
        self.half = kwargs.get("half", 0)

    def train(self):
        self.model.batch_size=self.batch_size
        if self.half:
            self.model = self.model.half()
        perturbed_model = model.PerturbedModel(self.model)
        ave_delta = .005 * self.batch_size
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay = 0, eps=1e-3)
        # opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=0)



        for epoch in range(self.epochs):
            print("Epoch:",epoch)
            total_reward = 0.0
            total_game_length = 0.0
            total_cards = 0.0
            total_points = 0.0
            for _ in trange(self.batches_per_epoch):
                perturbed_player = random.randrange(2)
                player_1, player_2 = (self.model, perturbed_model) if perturbed_player else (perturbed_model, self.model)
                init_score = 6 * random.randrange(2)
                top = random.randrange(2)

                with torch.no_grad():
                    perturbed_model.set_seed()
                    perturbed_model.perturb(self.noise_scale)

                    result, turns, cards, points = environment.run(player_1, player_2, size=self.batch_size, init_score=init_score, top=top)
                    if cuda_on:
                        result = result[perturbed_player].cuda().float()
                    else:
                        result = result[perturbed_player].float()
                    total_reward += result.mean()



                    total_game_length += turns.sum()
                    total_cards += cards.sum()
                    total_points += points.sum() - init_score * 2 * self.batch_size

                    reward_delta = result[self.batch_size//2:] - result[:self.batch_size//2]
                    step_size = reward_delta.view(directions,-1).sum(dim=1) / ((ave_delta + 1e-5) * self.noise_scale)
                    ave_delta = self.ave_delta_rate * ave_delta + (1 - self.ave_delta_rate) * (reward_delta.norm(p=1))
                    # step_size = reward_delta
                    perturbed_model.set_noise(self.noise_scale)
                perturbed_model.set_grad(step_size)
                # for param in self.model.parameters():
                #     if param.grad is not None:
                #         print(param.grad.abs().mean())
                self.noise_scale *= self.noise_scale_decay
                opt.step()
            # for param in self.model.parameters():
            #
            #     print(param.data.abs().mean())
            print("Average Reward:", total_reward / (self.batches_per_epoch))
            print("Average Game Length:", total_game_length.float() / (self.batches_per_epoch * self.batch_size))
            print("Average Cards:", total_cards.float() / (self.batches_per_epoch * self.batch_size))
            print("Average Points:", total_points.float() / (self.batches_per_epoch * self.batch_size))
            fname = os.path.join(self.checkpoints_dir, "epoch_"+str(epoch)+".pkl")
            perturbed_model.clear_noise()
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
    batch_size = 512
    directions = 64
    my_model = model.DenseNet(directions=directions)
    # Trainer(model.TransformerNet()).train()
    if cuda_on:
        my_model = my_model.cuda()
    Trainer(my_model,batch_size=batch_size,directions=directions).train()