import os
from collections import deque
import argparse
import tensorflow as tf
import numpy as np
import gym

from agent import DDPG


def main():
    env = gym.make('InvertedDoublePendulum-v1')

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    reward_history = deque(maxlen=100)

    def actor_network(states):
        h1_dim = 400
        h2_dim = 300

        # define policy neural network
        w1 = tf.get_variable("w1", [state_dim, h1_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", [h1_dim],
                       initializer=tf.constant_initializer(0))
        h1 = tf.nn.relu(tf.matmul(states, w1) + b1)

        w2 = tf.get_variable("w2", [h1_dim, h2_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", [h2_dim],
                       initializer=tf.constant_initializer(0))
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

        # use tanh to bound the action
        w3 = tf.get_variable("w3", [h2_dim, action_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable("b3", [action_dim],
                       initializer=tf.constant_initializer(0))

        # we assume actions range from [-1, 1]
        # you can scale action outputs with any constant here
        a = tf.nn.tanh(tf.matmul(h2, w3) + b3)
        return a

    def critic_network(states, action):
        h1_dim = 400
        h2_dim = 300

        # define policy neural network
        w1 = tf.get_variable("w1", [state_dim, h1_dim],
                           initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", [h1_dim],
                           initializer=tf.constant_initializer(0))
        h1 = tf.nn.relu(tf.matmul(states, w1) + b1)
        # skip action from the first layer
        h1_concat = tf.concat(1, [h1, action])

        w2 = tf.get_variable("w2", [h1_dim + action_dim, h2_dim],
                           initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", [h2_dim],
                           initializer=tf.constant_initializer(0))
        h2 = tf.nn.relu(tf.matmul(h1_concat, w2) + b2)

        w3 = tf.get_variable("w3", [h2_dim, 1],
                           initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable("b3", [1],
                           initializer=tf.constant_initializer(0))
        q = tf.matmul(h2, w3) + b3
        return q


    agent = DDPG(actor_network, critic_network, state_dim, action_dim)
    agent.construct_model(args.gpu)

    saver = tf.train.Saver(max_to_keep=1)
    if args.model_path is not None:
        # reuse saved model
        saver.restore(agent.sess, args.model_path)
    else:
        # build a new model
        agent.init_model()

    MAX_EPISODES = 10000
    MAX_STEPS = 1000

    for episode in xrange(MAX_EPISODES):
        # env init
        state = env.reset()
        env.render

        for step in xrange(MAX_STEPS):
            action = agent.sample_action(state[np.newaxis,:], explore=True)
            # act
            next_state, reward, done, _ = env.step(action)

            agent.store_experience(state, action, reward, next_state, done)

            agent.update_model()
            # shift
            state = next_state
            if done: break

        # Evaluation per 500 ep
        if episode % 500 == 0:
            total_steps = 0
            for ep_eval in xrange(100):
                total_rewards = 0
                state = env.reset()
                for step_eval in xrange(MAX_STEPS):
                    action = agent.sample_action(state[np.newaxis,:], explore=False)
                    next_state, reward, done, _ = env.step(action)
                    total_rewards += reward
                    state = next_state
                    if done:
                        break
                reward_history.append(total_rewards)

            mean_rewards = np.mean(reward_history)

            # logging
            print 'Ep%d ' % (episode)
            print 'Mean reward: %.2f' % mean_rewards
            if mean_rewards > 1900.0:
                print 'Problem solved!'
                print 'Saving model...'
                if not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                saver.save(agent.sess,
                    args.save_path + str(episode)+'_'+str(round(mean_rewards,2)))
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None, help=
            'Whether to use a saved model. (*None|model path)')
    parser.add_argument('--save_path', default='./model/', help=
            'Path to save a model during training.')
    parser.add_argument('--gpu', default=-1, help=
            'running on a specify gpu, -1 indicates using cpu')
    args = parser.parse_args()

    main()
