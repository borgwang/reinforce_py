import os
import time
import logger
import random
import tensorflow as tf
import gym
import numpy as np
from collections import deque

from config import args
from utils import set_global_seeds, sf01, explained_variance
from agent import PPO
from env_wrapper import make_env


def main():
    env = make_env()
    set_global_seeds(env, args.seed)

    agent = PPO(env=env)

    batch_steps = args.n_envs * args.batch_steps  # number of steps per update

    if args.save_interval and logger.get_dir():
        # some saving jobs
        pass

    ep_info_buffer = deque(maxlen=100)
    t_train_start = time.time()
    n_updates = args.n_steps // batch_steps
    runner = Runner(env, agent)

    for update in range(1, n_updates + 1):
        t_start = time.time()
        frac = 1.0 - (update - 1.0) / n_updates
        lr_now = args.lr  # maybe dynamic change
        clip_range_now = args.clip_range # maybe dynamic change
        obs, returns, masks, acts, vals, neglogps, advs, rewards, ep_infos = \
            runner.run(args.batch_steps, frac)
        ep_info_buffer.extend(ep_infos)
        loss_infos = []

        idxs = np.arange(batch_steps)
        for _ in range(args.n_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_steps, args.minibatch):
                end = start + args.minibatch
                mb_idxs = idxs[start: end]
                minibatch = [arr[mb_idxs] for arr in [obs, returns, masks, acts, vals, neglogps, advs]]
                loss_infos.append(agent.train(lr_now, clip_range_now, *minibatch))

        t_now = time.time()
        time_this_batch = t_now - t_start
        if update % args.log_interval == 0:
            ev = float(explained_variance(vals, returns))
            logger.logkv('updates', str(update) + '/' + str(n_updates))
            logger.logkv('serial_steps', update * args.batch_steps)
            logger.logkv('total_steps', update * batch_steps)
            logger.logkv('time', time_this_batch)
            logger.logkv('fps', int(batch_steps / (t_now - t_start)))
            logger.logkv('total_time', t_now - t_train_start)
            logger.logkv("explained_variance", ev)
            logger.logkv('avg_reward', np.mean([e['r'] for e in ep_info_buffer]))
            logger.logkv('avg_ep_len', np.mean([e['l'] for e in ep_info_buffer]))
            logger.logkv('adv_mean', np.mean(returns - vals))
            logger.logkv('adv_variance', np.std(returns - vals)**2)
            loss_infos = np.mean(loss_infos, axis=0)
            for loss_name, loss_info in zip(agent.loss_names, loss_infos):
                logger.logkv(loss_name, loss_info)
            logger.dumpkvs()

        if args.save_interval and update % args.save_interval == 0 and logger.get_dir():
            pass
    env.close()


class Runner:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.obs = np.zeros((args.n_envs,) + env.observation_space.shape, dtype=np.float32)
        self.obs[:] = env.reset()
        self.dones = [False for _ in range(args.n_envs)]

    def run(self, batch_steps, frac):
        b_obs, b_rewards, b_actions, b_values, b_dones, b_neglogps = [], [], [], [], [], []
        ep_infos = []

        for s in range(batch_steps):
            actions, values, neglogps = self.agent.step(self.obs, self.dones)
            b_obs.append(self.obs.copy())
            b_actions.append(actions)
            b_values.append(values)
            b_neglogps.append(neglogps)
            b_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeinfo = info.get('episode')
                if maybeinfo:
                    ep_infos.append(maybeinfo)
            b_rewards.append(rewards)
        # batch of steps to batch of rollouts
        b_obs = np.asarray(b_obs, dtype=self.obs.dtype)
        b_rewards = np.asarray(b_rewards, dtype=np.float32)
        b_actions = np.asarray(b_actions)
        b_values = np.asarray(b_values, dtype=np.float32)
        b_neglogps = np.asarray(b_neglogps, dtype=np.float32)
        b_dones = np.asarray(b_dones, dtype=np.bool)
        last_values = self.agent.get_value(self.obs, self.dones)

        b_returns = np.zeros_like(b_rewards)
        b_advs = np.zeros_like(b_rewards)
        lastgaelam = 0
        for t in reversed(range(batch_steps)):
            if t == batch_steps - 1:
                mask = 1.0 - self.dones
                nextvalues = last_values
            else:
                mask = 1.0 - b_dones[t + 1]
                nextvalues = b_values[t + 1]
            delta = b_rewards[t] + args.gamma * nextvalues * mask - b_values[t]
            b_advs[t] = lastgaelam = delta + args.gamma * args.lam * mask * lastgaelam
        b_returns = b_advs + b_values

        return (*map(sf01, (b_obs, b_returns, b_dones, b_actions, b_values, b_neglogps, b_advs, b_rewards)), ep_infos)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logger.configure()
    main()
