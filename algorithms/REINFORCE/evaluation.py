import argparse

import gym
from agent import REINFORCE
from utils import *


def main(args):

    def preprocess(obs):
        obs = obs[35:195]
        obs = obs[::2, ::2, 0]
        obs[obs == 144] = 0
        obs[obs == 109] = 0
        obs[obs != 0] = 1

        return obs.astype(np.float).ravel()

    INPUT_DIM = 80 * 80
    HIDDEN_UNITS = 200
    ACTION_DIM = 6

    # load agent
    agent = REINFORCE(INPUT_DIM, HIDDEN_UNITS, ACTION_DIM)
    agent.construct_model(args.gpu)

    # load model or init a new
    saver = tf.train.Saver(max_to_keep=1)
    if args.model_path is not None:
        # reuse saved model
        saver.restore(agent.sess, args.model_path)
    else:
        # build a new model
        agent.init_var()

    # load env
    env = gym.make('Pong-v0')

    # evaluation
    for ep in range(args.ep):
        # reset env
        total_rewards = 0
        state = env.reset()

        while True:
            env.render()
            # preprocess
            state = preprocess(state)
            # sample actions
            action = agent.sample_action(state[np.newaxis, :])
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
        '--ep', default=1, help='Test episodes')
    return parser.parse_args()


if __name__ == '__main__':
    main(args_parse())
