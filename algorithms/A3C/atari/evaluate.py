from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import time

from atari_env import Atari


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
        self.env = Atari(args, record=args.save_videos)
        self.global_net = global_net
        self.summary_writer = summary_writer
        self.global_steps_counter = global_steps_counter
        self.eval_every = args.eval_every
        self.eval_times = 0

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
                self.saver.save(sess, self.model_dir+str(global_steps))
                print('Model saved. Time cost: %.4fs ' %
                      (time.time() - save_start))

            time.sleep(self.eval_every)

    def _eval(self, sess):
        total_reward = 0.0
        episode_length = 0.0
        done = False
        for _ in range(3):
            s = self.env.new_round()
            while not done:
                p = sess.run(self.global_net.policy,
                             {self.global_net.inputs: [s]})
                a = np.random.choice(np.arange(self.env.a_dim), p=p[0])
                s, r, dead, done = self.env.step(a)
                total_reward += r
                episode_length += 1.0

        return total_reward / 3, episode_length / 3
