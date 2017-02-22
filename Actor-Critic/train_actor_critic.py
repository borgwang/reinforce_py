import os
import argparse
import gym
import numpy as np
import tensorflow as tf
from agent import ActorCritic

def main():
    INPUT_DIM = 80 * 80
    HIDDEN_UNITS = 200
    ACTION_DIM = 6
    MAX_EPISODES = 10000
    MAX_STEPS = 5000

    avg_win_steps = None

    def preprocess(obser):
        #preprocess 210x160x3 frame into 6400(80x80) flat vector
        obser = obser[35:195] # 160x160x3
        obser = obser[::2, ::2, 0] # downsample (80x80)
        obser[obser == 144] = 0
        obser[obser == 109] = 0
        obser[obser != 0] = 1

        return obser.astype(np.float).ravel()

    # load agent
    agent = ActorCritic(INPUT_DIM, HIDDEN_UNITS, ACTION_DIM)
    agent.construct_model(args.gpu)

    # load model or init a new
    saver = tf.train.Saver(max_to_keep=1)
    if args.model_path is not None:
        # reuse saved model
        saver.restore(agent.sess, MODEL_PATH)
        ep_base = int(args.model_path.split('_')[-1])
        mean_rewards = float(args.model_path.split('/')[-1].split('_')[0])
    else:
        # build a new model
        agent.init_var()
        ep_base = 0
        mean_rewards = None

    # load env
    env = gym.make("Pong-v0")

    # training loop
    for ep in xrange(MAX_EPISODES):
        # reset env
        step = 0
        total_rewards = 0
        state = preprocess(env.reset())

        while True:
            # sample actions
            action = agent.sample_action(state[np.newaxis,:])
            # act!
            next_state, reward, done, _ = env.step(action)

            next_state = preprocess(next_state)

            step += 1
            total_rewards += reward

            agent.store_rollout(state, action, reward)
            # state shift
            state = next_state

            if done: break

        # logging job
        # if ep % 10 == 0:
		# 	r = np.array(agent.reward_buffer,dtype=np.int)
		# 	index = np.where(r != 0)[0]
		# 	delta = np.roll(index, 1)
		# 	delta[0] = 0
		# 	steps_per_game = index - delta
		# 	result_per_game = r[r!=0]
		# 	win_index = np.where(result_per_game == 1)[0]
		# 	win_steps = np.mean(steps_per_game[win_index])
		# 	if avg_win_steps is None:
		# 		avg_win_steps = win_steps
		# 	else:
		# 		avg_win_steps = 0.99 * avg_win_steps + 0.01 * win_steps
		# 	print 'Average steps of win games: %.2f' % avg_win_steps

        if mean_rewards is None:
            mean_rewards = total_rewards
        else:
            mean_rewards = 0.99 * mean_rewards + 0.01 * total_rewards

        rounds = (21 - np.abs(total_rewards)) + 21
        average_steps = (step + 1) / rounds
        print 'Ep%s: %d rounds' % (ep_base+ep+1, rounds)
        print 'Average_steps: %.2f Reward: %s Average_reward: %.4f' % \
                (average_steps, total_rewards, mean_rewards)

        # update model per episode
        agent.update_model()
        # model saving
        if ep % 100 == 0: # model saving
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            saver.save(agent.sess,
                args.save_path+str(round(mean_rewards,2))+'_'+str(ep_base+ep+1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default=None,
            help='Whether to use a saved model. (*None|model path)')
    parser.add_argument('--save_path', default='./model/',
            help='Path to save a model during training.')
    parser.add_argument('--gpu', default=-1,
            help='running on a specify gpu, -1 indicates using cpu')
    args = parser.parse_args()

    main()
