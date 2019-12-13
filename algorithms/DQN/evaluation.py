import argparse

import gym
import tensorflow as tf

from agent import DQN


def main(args):
    # load env
    env = gym.make('CartPole-v0')
    # load agent
    agent = DQN(env)
    agent.construct_model(args.gpu)

    # load model or init a new
    saver = tf.train.Saver()
    if args.model_path is not None:
        # reuse saved model
        saver.restore(agent.sess, args.model_path)
    else:
        # build a new model
        agent.init_var()

    # training loop
    for ep in range(args.ep):
        # reset env
        total_rewards = 0
        state = env.reset()

        while True:
            env.render()
            # sample actions
            action = agent.sample_action(state, policy='greedy')
            # act!
            next_state, reward, done, _ = env.step(action)
            total_rewards += reward
            # state shift
            state = next_state
            if done:
                break

        print('Ep%s  Reward: %s ' % (ep+1, total_rewards))


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', default=None,
        help='Whether to use a saved model. (*None|model path)')
    parser.add_argument(
        '--gpu', default=-1,
        help='running on a specify gpu, -1 indicates using cpu')
    parser.add_argument(
        '--ep', type=int, default=1, help='Test episodes')
    return parser.parse_args()


if __name__ == '__main__':
    main(args_parse())
