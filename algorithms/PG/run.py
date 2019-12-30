import argparse

import numpy as np
import gym
import torch

from agent import VanillaPG
from agent import OffPolicyPG


def off_policy_run(env, args):
    agent = OffPolicyPG(env, args)
    global_steps = 0
    for ep in range(args.num_ep):
        rollouts, ep_steps, ep_rewards = run_episode(env, agent)
        global_steps += ep_steps
        agent.train(rollouts)
        ep_avg_rewards = np.mean(ep_rewards)
        print("Ep %d reward: %.4f ep_steps: %d global_steps: %d" % 
              (ep, ep_avg_rewards, ep_steps, global_steps))


def on_policy_run(env, args):
    agent = VanillaPG(env, args)
    global_steps = 0
    for ep in range(args.num_ep):
        rollouts, ep_steps, ep_rewards = run_episode(env, agent)
        global_steps += ep_steps
        agent.train(rollouts)
        ep_avg_rewards = np.mean(ep_rewards)
        print("Ep %d reward: %.4f ep_steps: %d global_steps: %d" % 
              (ep, ep_avg_rewards, ep_steps, global_steps))


def on_policy_run(env, args):
    agent = VanillaPG(env, args)
    global_steps = 0
    for ep in range(args.num_ep):
        rollouts, ep_steps, ep_rewards = run_episode(env, agent)
        global_steps += ep_steps
        agent.train(rollouts)
        ep_avg_rewards = np.mean(ep_rewards)
        print("Ep %d reward: %.4f ep_steps: %d global_steps: %d" % 
              (ep, ep_avg_rewards, ep_steps, global_steps))


def main(args):
    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    args.device = device
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # env and agent
    task_name = "BipedalWalker-v2"
    env = gym.make(task_name)
    env.seed(args.seed)
    # run
    # on_policy_run(env, args)
    off_policy_run(env, args)


def run_episode(env, agent):
    obs = env.reset()
    obs = preprocess(obs)
    ep_rewards, rollouts = [], []
    ep_steps = 0
    while True:
        logp, action = agent.step(obs)
        next_obs, reward, done, _ = env.step(action)
        ep_rewards.append(reward)
        next_obs = preprocess(next_obs)
        rollouts.append([obs, action, reward, logp, next_obs])
        obs = next_obs
        ep_steps += 1
        if done:
            break
    return rollouts, ep_steps, ep_rewards


def preprocess(obs):
    return obs.astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ep", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args())
