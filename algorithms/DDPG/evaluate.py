import argparse

import gym
import numpy as np
import tensorflow as tf
from gym import wrappers

from agent import DDPG


def main(args):
    env = gym.make('Walker2d-v1')
    env = wrappers.Monitor(env, './videos/', force=True)
    reward_history = []

    agent = DDPG(env, args)
    agent.construct_model(args.gpu)

    saver = tf.train.Saver()
    if args.model_path is not None:
        # reuse saved model
        saver.restore(agent.sess, args.model_path)
        ep_base = int(args.model_path.split('_')[-1])
        best_avg_rewards = float(args.model_path.split('/')[-1].split('_')[0])
    else:
        raise ValueError('model_path required!')

    for ep in range(args.ep):
        # env init
        state = env.reset()
        ep_rewards = 0
        for step in range(env.spec.timestep_limit):
            env.render()
            action = agent.sample_action(state[np.newaxis, :], noise=False)
            # act
            next_state, reward, done, _ = env.step(action[0])

            ep_rewards += reward
            agent.store_experience(state, action, reward, next_state, done)

            # shift
            state = next_state
            if done:
                break
        reward_history.append(ep_rewards)
        print('Ep%d  reward:%d' % (ep + 1, ep_rewards))

    print('Average rewards: ', np.mean(reward_history))


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', default='./models/',
        help='Whether to use a saved model. (*None|model path)')
    parser.add_argument(
        '--save_path', default='./models/',
        help='Path to save a model during training.')
    parser.add_argument(
        '--gpu', type=int, default=-1,
        help='running on a specify gpu, -1 indicates using cpu')
    parser.add_argument(
        '--seed', default=31, type=int, help='random seed')

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
    parser.add_argument('--ep', default=10, help='Test episodes')
    return parser.parse_args()


if __name__ == '__main__':
    main(args_parse())
