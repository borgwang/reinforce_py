import argparse
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from agent import DDPG


def main(args):
    set_random_seed(args.seed)
    env = gym.make('Walker2d-v1')
    agent = DDPG(env, args)
    agent.construct_model(args.gpu)

    saver = tf.train.Saver(max_to_keep=1)
    if args.model_path is not None:
        # reuse saved model
        saver.restore(agent.sess, args.model_path)
        ep_base = int(args.model_path.split('_')[-1])
        best_avg_rewards = float(args.model_path.split('/')[-1].split('_')[0])
    else:
        # build a new model
        agent.sess.run(tf.global_variables_initializer())
        ep_base = 0
        best_avg_rewards = None

    reward_history, step_history = [], []
    train_steps = 0

    for ep in range(args.max_ep):
        # env init
        state = env.reset()
        ep_rewards = 0
        for step in range(env.spec.timestep_limit):
            action = agent.sample_action(state[np.newaxis, :], noise=True)
            # act
            next_state, reward, done, _ = env.step(action[0])
            train_steps += 1
            ep_rewards += reward

            agent.store_experience(state, action, reward, next_state, done)
            agent.update_model()
            # shift
            state = next_state
            if done:
                print('Ep %d global_steps: %d Reward: %.2f' %
                      (ep + 1, agent.global_steps, ep_rewards))
                # reset ou noise
                agent.ou.reset()
                break
        step_history.append(train_steps)
        if not reward_history:
            reward_history.append(ep_rewards)
        else:
            reward_history.append(reward_history[-1] * 0.99 + ep_rewards + 0.01)

        # Evaluate during training
        if ep % args.log_every == 0 and ep > 0:
            ep_rewards = 0
            for ep_eval in range(args.test_ep):
                state = env.reset()
                for step_eval in range(env.spec.timestep_limit):
                    action = agent.sample_action(
                        state[np.newaxis, :], noise=False)
                    next_state, reward, done, _ = env.step(action[0])
                    ep_rewards += reward
                    state = next_state
                    if done:
                        break

            curr_avg_rewards = ep_rewards / args.test_ep

            # logging
            print('\n')
            print('Episode: %d' % (ep + 1))
            print('Global steps: %d' % agent.global_steps)
            print('Mean reward: %.2f' % curr_avg_rewards)
            print('\n')
            if not best_avg_rewards or (curr_avg_rewards >= best_avg_rewards):
                best_avg_rewards = curr_avg_rewards
                if not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                save_name = args.save_path + str(round(best_avg_rewards, 2)) \
                    + '_' + str(ep_base + ep + 1)
                saver.save(agent.sess, save_name)
                print('Model save %s' % save_name)

    plt.plot(step_history, reward_history)
    plt.xlabel('steps')
    plt.ylabel('running reward')
    plt.show()


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', default=None,
        help='Whether to use a saved model. (*None|model path)')
    parser.add_argument(
        '--save_path', default='./models/',
        help='Path to save a model during training.')
    parser.add_argument(
        '--log_every', default=100,
        help='Interval of logging and may be model saving')
    parser.add_argument(
        '--gpu', type=int, default=-1,
        help='running on a specify gpu, -1 indicates using cpu')
    parser.add_argument(
        '--seed', default=31, type=int, help='random seed')

    parser.add_argument(
        '--max_ep', type=int, default=10000, help='Number of training episodes')
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
        help='soft target update rate')
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


def set_random_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == '__main__':
    main(args_parse())
