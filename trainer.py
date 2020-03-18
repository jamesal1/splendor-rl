from tqdm import trange
import os
import model
import datetime
import shutil
import torch
import environment
import random
random.seed(0)
torch.manual_seed(0)


cuda_device = torch.device("cuda")

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
        self.noise_scale = kwargs.get("noise_scale", 1e-3)
        # self.noise_scale_decay = 1 - kwargs.get("noise_scale_decay", 1e-3)
        self.noise_scale_decay = 1 - kwargs.get("noise_scale_decay", 0)
        self.lr = kwargs.get("lr", 1e-3)
        # self.lr_decay = kwargs.get("lr_decay", 1e-2)
        self.ave_delta_rate = kwargs.get("ave_delta_rate", .999)
        self.epochs = kwargs.get("epochs", 1000)
        self.batches_per_epoch = kwargs.get("batches_per_epoch",100)
        self.batch_size = kwargs.get("batch_size",128)

    def train(self):
        perturbed_model = type(self.model)().cuda()
        noise_model = type(self.model)().cuda()
        ave_delta = .1
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # opt = torch.optim.SGD(self.model.parameters(), lr=self.lr)



        for epoch in range(self.epochs):
            total_reward = 0
            total_game_length = 0
            total_points = 0
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

                    add_result, add_turns, add_points = environment.run(player_1, player_2, size=self.batch_size, init_score=init_score, top=top)

                    add_score = add_result[perturbed_player].sum()/self.batch_size
                    for p_param, n_param in zip(perturbed_model.parameters(), noise_model.parameters()):
                        p_param.sub_(2 * n_param.data)

                    sub_result, sub_turns, sub_points = environment.run(player_1, player_2, size=self.batch_size, init_score=init_score, top=top)
                    sub_score = sub_result[perturbed_player].sum()/self.batch_size
                    total_reward += add_score + sub_score

                    total_game_length += add_turns.sum() + sub_turns.sum()
                    total_points += add_points + sub_points
                    reward_delta = sub_score - add_score

                    step_size = reward_delta / (ave_delta + 1e-5)
                    ave_delta = self.ave_delta_rate * ave_delta + (1 - self.ave_delta_rate) * abs(reward_delta)

                for param, n_param in zip(self.model.parameters(), noise_model.parameters()):
                    param.grad = (step_size * n_param.data)

                self.noise_scale *= self.noise_scale_decay
                opt.step()
            print("Average Reward:", total_reward / (2 * self.batches_per_epoch))
            print("Average Game Length:", total_game_length.float() / (2 * self.batches_per_epoch * self.batch_size))
            print("Average Points:", total_points.float() / (2 * self.batches_per_epoch * self.batch_size))
            fname = os.path.join(self.checkpoints_dir,"epoch_"+str(epoch)+".pkl")
            torch.save(self.model, fname)




def no_grad_test():
    import time
    for i in trange(100):
        time.sleep(1)

    var = torch.eye(5)
    var.grad = torch.ones((5, 5))

    print(var)



if __name__ == "__main__":
    Trainer(model.DenseNet().cuda()).train()