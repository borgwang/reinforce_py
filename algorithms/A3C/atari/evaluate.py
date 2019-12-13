import os
import time

import numpy as np
import tensorflow as tf

from atari_env import A_DIM
from atari_env import make_env


class Evaluate(object):
    '''
    Evaluate a policy by running n episodes in an environment.
    Save a video and plot summaries to Tensorboard

    Args:
        global_net: The global network
        summary_writer: used to write Tensorboard summaries
        args: Some global parameters
    '''

    def __init__(self, global_net, summary_writer, global_steps_counter, args):
        self.env = make_env(args, record_video=args.record_video)
        self.global_net = global_net
        self.summary_writer = summary_writer
        self.global_steps_counter = global_steps_counter
        self.eval_every = args.eval_every
        self.eval_times = 0
        self.eval_episodes = args.eval_episodes

        self.saver = tf.train.Saver(max_to_keep=5)
        self.model_dir = os.path.join(args.save_path, 'models/')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def run(self, sess, coord):
        while not coord.should_stop():
            global_steps = next(self.global_steps_counter)
            eval_start = time.time()
            avg_reward, avg_ep_length = self._eval(sess)
            self.eval_times += 1
            print('Eval at step %d: avg_reward %.4f, avg_ep_length %.4f' %
                  (global_steps, avg_reward, avg_ep_length))
            print('Time cost: %.4fs' % (time.time() - eval_start))
            # add summaries
            ep_summary = tf.Summary()
            ep_summary.value.add(
                simple_value=avg_reward, tag='eval/avg_reward')
            ep_summary.value.add(
                simple_value=avg_ep_length, tag='eval/avg_ep_length')
            self.summary_writer.add_summary(ep_summary, global_steps)
            self.summary_writer.flush()
            # save models
            if self.eval_times % 10 == 1:
                save_start = time.time()
                self.saver.save(sess, self.model_dir + str(global_steps))
                print('Model saved. Time cost: %.4fs ' %
                      (time.time() - save_start))

            time.sleep(self.eval_every)

    def _eval(self, sess):
        total_reward = 0.0
        episode_length = 0.0
        for _ in range(self.eval_episodes * 5):
            s = self.env.reset()
            while True:
                p = sess.run(self.global_net.policy,
                             {self.global_net.inputs: [s]})
                a = np.random.choice(range(A_DIM), p=p[0])
                s, r, done, _ = self.env.step(a)
                total_reward += r
                episode_length += 1.0
                if done:
                    break
        return total_reward / self.eval_episodes, \
            episode_length / self.eval_episodes
