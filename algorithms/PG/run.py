import argparse
from collections import namedtuple

import numpy as np
import gym
import torch
import torch.distributions as tdist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, out_dim)

        self.logstd = torch.tensor(np.ones(out_dim, dtype=np.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)
        std = torch.exp(self.logstd)
        dist = tdist.Normal(mean, std)
        out = dist.sample()
        logp = dist.log_prob(out)
        return logp, out


class VanillaPG:

    def __init__(self, env):
        self.obs_space = env.observation_space
        self.act_space = env.action_space

        # build policy net
        self.policy = Net(in_dim=self.obs_space.shape[0],
                          out_dim=self.act_space.shape[0])
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

    def step(self, obs):
        if obs.ndim == 1:
            obs = obs.reshape((1, -1))
        obs = torch.tensor(obs, requires_grad=True)
        with torch.no_grad():
            logp, out = self.policy(obs)
        return logp, out

    def train(self, traj):
        obs, acts, rewards, logp = np.array(traj).T
        # calculate return estimations
        logp = torch.cat(list(logp))
        acts = torch.cat(list(acts))
        returns = np.array([np.sum(rewards[i:]) for i in range(len(rewards))])
        returns = torch.tensor(returns.reshape((-1, 1)))
        grads = (logp * returns)
        # buggy
        acts.backward(grads)


def main(args):
    task_name = "BipedalWalker-v2"
    env = gym.make(task_name)
    agent = VanillaPG(env)

    for ep in range(args.num_ep):
        obs = env.reset()
        ep_rewards, trajactory = [], []
        while True:
            obs = preprocess(obs)
            logp, action = agent.step(obs)
            next_obs, reward, done, _ = env.step(action[0].numpy())
            ep_rewards.append(reward)
            trajactory.append([obs, action, reward, logp])
            obs = next_obs
            if done:
                break
        agent.train(trajactory)
        print("Ep %d reward: %.4f" % (ep, np.mean(ep_rewards)))
            

def preprocess(obs):
    obs = obs.astype(np.float32)
    return obs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ep", type=int, default=5000)
    main(parser.parse_args())



