from collections import deque
import random

import numpy as np
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Backbone(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 400)
        self.fc2 = nn.Linear(400, 200)

        self.bn1 = nn.BatchNorm1d(400)
        self.bn2 = nn.BatchNorm1d(200)


class PolicyNet(Backbone):

    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim)
<<<<<<< Updated upstream
        self.out_dim = out_dim

        self.fc3 = nn.Linear(200, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.fc4 = nn.Linear(100, self.out_dim)

        self.logstd = torch.tensor(np.zeros((1, out_dim), dtype=np.float32),
=======
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, out_dim)

        self.logstd = torch.tensor(np.ones(out_dim, dtype=np.float32),
>>>>>>> Stashed changes
            requires_grad=True).to(args.device)

    def get_dist(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
<<<<<<< Updated upstream
        x = self.bn3(x)
        mean = self.fc4(x)
        std = torch.exp(self.logstd)
        randn = torch.randn(mean.size()).to(args.device)
        out = randn * std + mean
        logp = -torch.log(std * (2 * np.pi) ** 0.5) + (out - mean) ** 2 / (2 * std ** 2)
        print(next(self.fc4.parameters())[0][:5])
=======
        mean = self.fc4(x)
        std = torch.exp(self.logstd)
        dist = tdist.Normal(mean, std)
        return dist

    def forward(self, x):
        dist = self.get_dist(x)
        out = dist.sample()
        logp = dist.log_prob(out)
>>>>>>> Stashed changes
        return logp, out

    def get_logp(self, s, a):
        dist = self.get_dist(s)
        logp = dist.log_prob(a)
        return logp

class ValueNet(Backbone):

    def __init__(self, in_dim):
        super().__init__(in_dim)
        self.fc3 = nn.Linear(200, 10)
<<<<<<< Updated upstream
        self.bn3 = nn.BatchNorm1d(10)
=======
>>>>>>> Stashed changes
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
<<<<<<< Updated upstream
        x = self.bn3(x)
=======
>>>>>>> Stashed changes
        x = self.fc4(x)
        return x


class VanillaPG:

    def __init__(self, env, args_):
        global args
        args = args_
        self.obs_space = env.observation_space
        self.act_space = env.action_space

        self._build_nets()

    def step(self, obs):
        if obs.ndim == 1:
            obs = obs.reshape((1, -1))
        obs = torch.tensor(obs).to(args.device)
        self.policy.eval()
        logp, out = self.policy(obs)
        out = out[0].detach().cpu().numpy()
        return logp, out

    def train(self, traj):
        obs, acts, rewards, logp, next_obs = np.array(traj).T

        obs, next_obs = np.stack(obs), np.stack(next_obs)
        obs_combine = np.concatenate([obs, next_obs], axis=0)
        obs_combine = torch.tensor(obs_combine).to(args.device)

        self.value_func.eval()
        vs_combine = self.value_func(obs_combine)
        vs_combine_np = vs_combine.cpu().detach().numpy().flatten()

        vs_np = vs_combine_np[:len(vs_combine)//2]
        next_vs_np = vs_combine_np[len(vs_combine)//2:]
        vs = vs_combine[:len(vs_combine)//2]

        logp = torch.cat(logp.tolist())

        # calculate return estimations
        phi = self._calculate_phi(rewards, vs_np, next_vs_np)

        # update policy parameters
        self.policy.train()
        self.policy_optim.zero_grad()
        logp.backward(phi)
        self.policy_optim.step()

        # update value functions
        self.value_func.train()
        rwd_to_go = [np.sum(rewards[i:]) for i in range(len(rewards))]
        target = torch.tensor(rwd_to_go).view((-1, 1)).to(args.device)
        vf_loss = F.mse_loss(vs, target)
        print("vf mse: %.4f" % vf_loss)
        self.vf_optim.zero_grad()
        vf_loss.backward()
        self.vf_optim.step()

    def _build_nets(self):
        policy = PolicyNet(in_dim=self.obs_space.shape[0],
                           out_dim=self.act_space.shape[0])
        value_func = ValueNet(in_dim=self.obs_space.shape[0])
        self.policy = policy.to(args.device)
        self.value_func = value_func.to(args.device)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.vf_optim = optim.Adam(self.value_func.parameters(), lr=args.lr)

    def _discount(self, arr, alpha=0.99):
        discount_arr = []
        for a in arr:
            discount_arr.append(alpha * a)
            alpha *= alpha
        return discount_arr

    def _calculate_phi(self, rewards, vs, next_vs):
        # option1: raw returns
        raw_returns = [np.sum(rewards) for i in range(len(rewards))]
        # option2: rewards to go
        rwd_to_go = [np.sum(rewards[i:]) for i in range(len(rewards))]
        # option3: discounted rewards to go
        disc_rwd_to_go = [np.sum(self._discount(rewards[i:])) 
                          for i in range(len(rewards))]
        # option4: td0 estimation
        td0 = rewards + next_vs
        # subtract baseline
        baseline = vs

        phi = disc_rwd_to_go - vs
        phi = np.array(phi).reshape((-1, 1))
        phi = np.tile(phi, (1, 4))
        phi = torch.tensor(phi).to(args.device)
        return phi


class OffPolicyPG(VanillaPG):

    def __init__(self, env, args_):
        super().__init__(env, args_)
        self.memory = deque(maxlen=10000)

    def train(self, traj):
        obs, acts, rewards, logp, next_obs = np.array(traj).T
        disc_rwd_to_go = np.array([np.sum(self._discount(rewards[i:])) 
                          for i in range(len(rewards))])
        for i in range(len(obs)):
            data_tuple = [obs[i], acts[i], rewards[i], disc_rwd_to_go[i], 
                          logp[i], next_obs[i]]
            self.memory.append(data_tuple)

        if len(self.memory) < 10000:
            print("reply memory not warm up")
            return
        traj = random.sample(self.memory, 1024)
        obs, acts, rewards, disc_rwd, logp, next_obs = np.array(traj).T

        disc_rwd = disc_rwd.astype(np.float32)
        acts = np.stack(acts)
        obs, next_obs = np.stack(obs), np.stack(next_obs)
        obs_combine = np.concatenate([obs, next_obs], axis=0)
        obs_combine = torch.tensor(obs_combine).to(args.device)

        self.value_func.eval()
        vs_combine = self.value_func(obs_combine)
        vs_combine_np = vs_combine.cpu().detach().numpy().flatten()

        vs_np = vs_combine_np[:len(vs_combine)//2]
        next_vs_np = vs_combine_np[len(vs_combine)//2:]
        vs = vs_combine[:len(vs_combine)//2]

        old_logp = torch.cat(logp.tolist())
        new_logp = self.policy.get_logp(torch.tensor(obs).to(args.device), 
                                        torch.tensor(acts).to(args.device))
        ratio = torch.exp(new_logp - old_logp)
        ratio = torch.clamp(ratio, 0.9, 1.1)

        # calculate return estimations
        phi = disc_rwd - vs_np
        phi = np.array(phi).reshape((-1, 1))
        phi = np.tile(phi, (1, 4))
        phi = torch.tensor(phi).to(args.device)

        # update policy parameters
        self.policy.train()
        self.policy_optim.zero_grad()
        new_logp.backward(phi * ratio)
        self.policy_optim.step()

        # update value functions
        self.value_func.train()
        target = torch.tensor(disc_rwd).view((-1, 1)).to(args.device)
        vf_loss = F.mse_loss(vs, target)
        print("vf mse: %.4f" % vf_loss)
        self.vf_optim.zero_grad()
        vf_loss.backward()
        self.vf_optim.step()
