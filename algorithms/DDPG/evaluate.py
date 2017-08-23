import argparse
import tensorflow as tf
import numpy as np
import gym

from agent import DDPG


def main(args):
    env = gym.make('Walker2d-v1')

    reward_history = []

    agent = DDPG(env)
    agent.construct_model(args.gpu)

    saver = tf.train.Saver()
    if args.model_path is not None:
        # reuse saved model
        saver.restore(agent.sess, args.model_path)
    else:
        # build a new model
        agent.sess.run(tf.global_variables_initializer())

    for episode in xrange(args.ep):
        # env init
        state = env.reset()
        total_rewards = 0
        for step in xrange(env.spec.timestep_limit):
            env.render()
            action = agent.sample_action(state[np.newaxis, :], explore=False)
            # act
            next_state, reward, done, _ = env.step(action[0])

            total_rewards += reward
            agent.store_experience(state, action, reward, next_state, done)

            agent.update_model()
            # shift
            state = next_state
            if done:
                break
        reward_history.append(total_rewards)
        print 'Ep%d  reward:%d' % (episode+1, total_rewards)

    print 'Average rewards: ', np.mean(reward_history)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None,
                        help='Whether to use a saved model. (*None|model path)')
    parser.add_argument('--save_path', default='./model/',
                        help='Path to save a model during training.')
    parser.add_argument('--gpu', default=-1,
                        help='running on a specify gpu, -1 indicates using cpu')
    parser.add_argument('--ep', default=10, help='Test episodes')
    return parser.parse_args()


if __name__ == '__main__':
    main(args_parse())
