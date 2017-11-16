from __future__ import print_function
from __future__ import division

import gym
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt

from agent import Agent


def main(args):
    env = gym.make('Pendulum-v0').unwrapped
    agent = Agent(env, args)
    reward_history = []
    start_time = time.time()
    # main loop
    for ep in range(args.max_ep):
        buffer_s, buffer_a, buffer_r = [], [], []
        s = env.reset()
        ep_reward = 0
        for t in range(args.ep_len):
            # env.render()
            a = agent.sample_action(s)
            next_s, r, done, _ = env.step(a)
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r + 8) / 8)
            s = next_s
            ep_reward += r

            # update agent
            if (t + 1) % args.batch_size == 0 or t == args.ep_len - 1:
                next_s_value = agent.get_value(next_s)
                #  calculate discounted rewards
                discounted_r = []
                for r in buffer_r[::-1]:
                    next_s_value = r + args.gamma * next_s_value
                    discounted_r.append(next_s_value)
                discounted_r.reverse()
                b_s = np.vstack(buffer_s)
                b_a = np.vstack(buffer_a)
                b_r = np.asarray(discounted_r)[:, np.newaxis]
                buffer_s, buffer_a, buffer_r = [], [], []
                agent.update_model(b_s, b_a, b_r)

        if ep == 0:
            reward_history.append(ep_reward)
        else:
            reward_history.append(reward_history[-1] * 0.99 + ep_reward * 0.01)

        print('Ep %d  reward: %d' % (ep, ep_reward))

    print('train finished. time cost: %.4fs' % (time.time() - start_time))
    plt.plot(np.arange(len(reward_history)), reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig('result.png')


def args_parse():
        parser = argparse.ArgumentParser()
        parser.add_argument('--max_ep', type=int, default=1000,
                            help='Max training episodes.')
        parser.add_argument('--ep_len', type=int, default=200,
                            help='Max steps of an episode.')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--actor_lr', type=float, default=0.0001,
                            help='learning rate of actor')
        parser.add_argument('--critic_lr', type=float, default=0.0002,
                            help='learning rate of critic')
        parser.add_argument('--a_update_steps', type=int, default=10)
        parser.add_argument('--c_update_steps', type=int, default=20)
        parser.add_argument('--method', default='clip',
                            help='kl_pen or clip')
        parser.add_argument('--lamb', type=float, default=0.5,
                            help='hyperparameter of kl penalty method')
        parser.add_argument('--kl_target', type=float, default=0.01,
                            help='hyperparameter of kl penalty method')
        parser.add_argument('--epsilon', type=float, default=0.2,
                            help='hyperparameter of clip method')
        parser.add_argument('--gamma', type=float, default=0.99,
                            help='Discounted factor')
        parser.add_argument('--gpu', type=int, default=-1,
                            help='running on a specify gpu, -1 indicates cpu')
        return parser.parse_args()


if __name__ == '__main__':
    main(args_parse())
