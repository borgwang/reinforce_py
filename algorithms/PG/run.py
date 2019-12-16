import argparse
from collections import namedtuple

import numpy as np
import gym
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

        self.logstd = torch.tensor(np.ones(out_dim, dtype=np.float32),
            requires_grad=True).to(args.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VanillaPG:

    def __init__(self, env):
        self.obs_space = env.observation_space
        self.act_space = env.action_space

        self._build_nets()

    def step(self, obs):
        if obs.ndim == 1:
            obs = obs.reshape((1, -1))
        obs = torch.tensor(obs).to(args.device)
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
        self.vf_optim = optim.Adam(self.value_func.parameters(), lr=args.lr*5)

    def train(self, traj):
        obs, acts, rewards, logp = np.array(traj).T
        obs = torch.tensor(np.stack(obs)).to(args.device)

        vs = self.value_func(obs)
        vs_numpy = vs.cpu().detach().numpy().flatten()

        logp = torch.cat(logp.tolist())

        # calculate return estimations
        reward_to_go = [np.sum(rewards[i:]) for i in range(len(rewards))]

        ret = [reward_to_go[i] - vs_numpy[i] for i in range(len(rewards))]
        ret = np.reshape(ret, (-1, 1))
        ret = np.tile(ret, (1, 4))
        ret = torch.tensor(ret).to(args.device)

        # update policy parameters
        self.policy_optim.zero_grad()
        logp.backward(ret)
        self.policy_optim.step()

        # update value functions
        target = torch.tensor(reward_to_go).view((-1, 1)).to(args.device)
        vf_loss = F.mse_loss(vs, target)
        print("vf mse: %.4f" % vf_loss)
        self.vf_optim.zero_grad()
        vf_loss.backward()
        self.vf_optim.step()


def main():
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    args.device = device

    # env and agent
    task_name = "BipedalWalker-v2"
    env = gym.make(task_name)
    agent = VanillaPG(env)

    ep_running_rewards = 0.0
    global_steps = 0
    for ep in range(args.num_ep):
        obs = env.reset()
        ep_rewards, trajactory = [], []
        ep_steps = 0
        while True:
            obs = preprocess(obs)
            logp, action = agent.step(obs)
            next_obs, reward, done, _ = env.step(action)
            ep_rewards.append(reward)
            trajactory.append([obs, action, reward, logp])
            obs = next_obs
            ep_steps += 1
            if done:
                break
        global_steps += ep_steps
        agent.train(trajactory)
        ep_avg_rewards = np.mean(ep_rewards)
        ep_running_rewards = 0.1 * ep_avg_rewards + 0.9 * ep_running_rewards
        print("Ep %d reward: %.4f ep_steps: %d global_steps: %d" % 
              (ep, ep_running_rewards, ep_steps, global_steps))
            

def preprocess(obs):
    obs = obs.astype(np.float32)
    return obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ep", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--gpu", action="store_true")
    global args
    args = parser.parse_args()
    main()



