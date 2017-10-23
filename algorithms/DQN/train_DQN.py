from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import argparse
import gym
import tensorflow as tf

from agent import DQN


def main(args):
    # Hyper parameters
    MAX_EPISODE = 10000  # training episode
    INITIAL_EPSILON = 0.5   # starting value of epsilon
    FINAL_EPSILON = 0.01    # final value of epsilon
    TEST_EPISODE = 100

    env = gym.make('CartPole-v0')
    agent = DQN(env, double_q=args.double)
    agent.construct_model(args.gpu)

    saver = tf.train.Saver(max_to_keep=2)
    if args.model_path is not None:
        saver.restore(agent.sess, args.model_path)
        ep_base = int(args.model_path.split('_')[-1])
        mean_rewards = float(args.model_path.split('/')[-1].split('_')[0])
    else:
        agent.sess.run(tf.global_variables_initializer())
        ep_base = 0
        mean_rewards = None

    # Training
    for ep in range(MAX_EPISODE):
        state = env.reset()

        for step in range(env.spec.timestep_limit):
            # pick action
            action = agent.sample_action(state, policy='egreedy')
            # Execution action.
            next_state, reward, done, debug = env.step(action)
            # modified reward to speed up learning
            reward = 0.1 if not done else -1
            # Learn and Update net parameters
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            if done:
                break

        # Update epsilon
        if agent.epsilon > FINAL_EPSILON:
            agent.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / MAX_EPISODE

        # Evaluate during training
        if ep % args.log_every == args.log_every-1:
            total_reward = 0
            for i in range(TEST_EPISODE):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    action = agent.sample_action(state, policy='greedy')
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            mean_rewards = total_reward / float(TEST_EPISODE)
            print('Episode:', ep+1, ' Average Reward:', mean_rewards)
            print('Global steps:', agent.global_step)

            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            save_name = args.save_path + str(round(mean_rewards, 2)) + '_' \
                + str(ep_base+ep+1)
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
        '--double', default=False, help='enable or disable double dqn')
    parser.add_argument(
        '--log_every', default=100, help='Log and save model every x episodes')
    parser.add_argument(
        '--gpu', default=-1,
        help='running on a specify gpu, -1 indicates using cpu')
    return parser.parse_args()


if __name__ == '__main__':
    main(args_parse())
