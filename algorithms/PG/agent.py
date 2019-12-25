import numpy as np
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyNet(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, out_dim)

        self.bn1 = nn.BatchNorm1d(400)
        self.bn2 = nn.BatchNorm1d(200)

        self.logstd = torch.tensor(np.ones(out_dim, dtype=np.float32),
            requires_grad=True).to(args.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        mean = self.fc3(x)
        std = torch.exp(self.logstd)
        dist = tdist.Normal(mean, std)
        out = dist.sample()
        logp = dist.log_prob(out)
        return logp, out


class ValueNet(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 1)

        self.bn1 = nn.BatchNorm1d(400)
        self.bn2 = nn.BatchNorm1d(200)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
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
        out = out[0].cpu().numpy()
        return logp, out

    def _build_nets(self):
        policy = PolicyNet(in_dim=self.obs_space.shape[0],
                           out_dim=self.act_space.shape[0])
        value_func = ValueNet(in_dim=self.obs_space.shape[0])
        self.policy = policy.to(args.device)
        self.value_func = value_func.to(args.device)

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.vf_optim = optim.Adam(self.value_func.parameters(), lr=args.lr)

    def train(self, traj):
        obs, acts, rewards, logp, next_obs = np.array(traj).T
        obs, next_obs = np.stack(obs), np.stack(next_obs)
        obs = np.concatenate([obs, next_obs], axis=0)
        obs = torch.tensor(obs).to(args.device)

        self.value_func.eval()
        vs = self.value_func(obs)
        vs_numpy = vs.cpu().detach().numpy().flatten()
        vs_numpy, next_vs_numpy = vs_numpy[:len(vs)//2], vs_numpy[len(vs)//2:]
        vs = vs[:len(vs)//2]

        logp = torch.cat(logp.tolist())

        reward_to_go = [np.sum(self.discount(rewards[i:])) 
                        for i in range(len(rewards))]

        # calculate return estimations
        ret = reward_to_go
        # ret = [rewards[i] + next_vs_numpy[i] - vs_numpy[i] 
        #        for i in range(len(rewards))]
        ret = [reward_to_go[i] - vs_numpy[i] for i in range(len(rewards))]

        ret = np.array(ret)
        ret = np.reshape(ret, (-1, 1))
        ret = np.tile(ret, (1, 4))
        ret = torch.tensor(ret).to(args.device)

        # update policy parameters
        self.policy.train()
        self.policy_optim.zero_grad()
        logp.backward(ret)
        self.policy_optim.step()

        # update value functions
        self.value_func.train()
        # td = [rewards[i] + next_vs_numpy[i] for i in range(len(rewards))]
        target = torch.tensor(reward_to_go).view((-1, 1)).to(args.device)
        vf_loss = F.mse_loss(vs, target)
        print("vf mse: %.4f" % vf_loss)
        self.vf_optim.zero_grad()
        vf_loss.backward()
        self.vf_optim.step()

    def discount(self, arr, alpha=0.99):
        discount_arr = []
        for a in arr:
            discount_arr.append(alpha * a)
            alpha *= alpha
        return discount_arr
