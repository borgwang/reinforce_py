import argparse
import gym
import numpy as np
import tensorflow as tf
from agent import ActorCritic

def main():
    def preprocess(obser):
        obser = obser[35:195] # 160x160x3
        obser = obser[::2, ::2, 0] # downsample (80x80)
        obser[obser == 144] = 0
        obser[obser == 109] = 0
        obser[obser != 0] = 1

        return obser.astype(np.float).ravel()

    INPUT_DIM = 80 * 80
    HIDDEN_UNITS = 200
    ACTION_DIM = 6

    # load agent
    agent = ActorCritic(INPUT_DIM, HIDDEN_UNITS, ACTION_DIM)
    agent.construct_model(args.gpu)

    # load model or init a new
    saver = tf.train.Saver(max_to_keep=1)
    if args.model is not None:
        # reuse saved model
        saver.restore(agent.sess, args.model)
    else:
        # build a new model
        agent.init_var()

    # load env
    env = gym.make("Pong-v0")

    # training loop
    for ep in xrange(args.ep):
        # reset env
        total_rewards = 0
        state = env.reset()

        while True:
            env.render()
            # preprocess
            state = preprocess(state)
            # sample actions
            action = agent.sample_action(state[np.newaxis,:])
            # act!
            next_state, reward, done, _ = env.step(action)
            total_rewards += reward
            # state shift
            state = next_state
            if done: break

        print 'Ep%s  Reward: %s ' % (ep+1, total_rewards)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./model/12.31_12000', help=
            'Whether to use a saved model. (*None|model path)')
    parser.add_argument('--gpu', default=-1, help=
            'running on a specify gpu, -1 indicates using cpu')
    parser.add_argument('--ep', default=1, help=
            'Test episodes')
    args = parser.parse_args()
    main()
