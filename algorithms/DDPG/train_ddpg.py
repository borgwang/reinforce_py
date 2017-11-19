from __future__ import print_function
from __future__ import division

import os
import argparse
import tensorflow as tf
import numpy as np
import gym

from agent import DDPG


def main(args):
    env = gym.make('Walker2d-v1')

    agent = DDPG(env, args)
    agent.construct_model(args.gpu)

    saver = tf.train.Saver(max_to_keep=1)
    if args.model_path is not None:
        # reuse saved model
        saver.restore(agent.sess, args.model_path)
        ep_base = int(args.model_path.split('_')[-1])
    else:
        # build a new model
        agent.sess.run(tf.global_variables_initializer())
        ep_base = 0

    for episode in range(args.max_ep):
        # env init
        state = env.reset()
        total_rewards = 0
        for step in range(env.spec.timestep_limit):
            action = agent.sample_action(state[np.newaxis, :], explore=True)
            # act
            next_state, reward, done, _ = env.step(action[0])
            total_rewards += reward

            agent.store_experience(state, action, reward, next_state, done)

            agent.update_model()
            # shift
            state = next_state
            if done:
                print('Ep %d global_steps: %d Reward: %.2f' %
                      (episode+1, agent.global_steps, total_rewards))
                # reset ou noise
                agent.ou.reset()
                break

        # Evaluation per 100 ep
        if episode % 100 == 0 and episode > 100:
            total_rewards = 0
            for ep_eval in range(args.test_ep):
                state = env.reset()
                for step_eval in range(env.spec.timestep_limit):
                    action = agent.sample_action(
                        state[np.newaxis, :], explore=False)
                    next_state, reward, done, _ = env.step(action[0])
                    total_rewards += reward
                    state = next_state
                    if done:
                        break

            mean_rewards = total_rewards / args.test_ep

            # logging
            print('\n')
            print('Episode: %d' % (episode + 1))
            print('Gloabal steps: %d' % agent.global_steps)
            print('Mean reward: %.2f' % mean_rewards)
            print('\n')
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            save_name = args.save_path \
                + str(episode) + '_' + str(round(mean_rewards, 2))
            saver.save(agent.sess, save_name)


def args_parse():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--model_path', default=None,
            help='Whether to use a saved model. (*None|model path)')
        parser.add_argument(
            '--save_path', default='./model/',
            help='Path to save a model during training.')
        parser.add_argument(
            '--gpu', default=-1,
            help='running on a specify gpu, -1 indicates using cpu')

        parser.add_argument(
            '--max_ep', type=int, default=100000, help='Number of training episodes')
        parser.add_argument(
            '--test_ep', type=int, default=10, help='Number of test episodes')
        parser.add_argument(
            '--a_lr', type=float, default=1e-4, help='Actor learning rate')
        parser.add_argument(
            '--c_lr', type=float, default=1e-3, help='Critic learning rate')
        parser.add_argument(
            '--batch_size', type=int, default=64, help='Size of training batch')
        parser.add_argument(
            '--gamma', type=float, default=0.99, help='Discounted factor')
        parser.add_argument(
            '--target_update_rate', type=float, default=0.001,
            help='parameter of soft target update')
        parser.add_argument(
            '--reg_param', type=float, default=0.01, help='l2 regularization')
        parser.add_argument(
            '--buffer_size', type=int, default=1000000, help='Size of memory buffer')
        parser.add_argument(
            '--replay_start_size', type=int, default=1000,
            help='Number of steps before learning from replay memory')
        parser.add_argument(
            '--noise_theta', type=float, default=0.15,
            help='Ornstein-Uhlenbeck noise parameters')
        parser.add_argument(
            '--noise_sigma', type=float, default=0.20,
            help='Ornstein-Uhlenbeck noise parameters')
        return parser.parse_args()


if __name__ == '__main__':
    main(args_parse())
