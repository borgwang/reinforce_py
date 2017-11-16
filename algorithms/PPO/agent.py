from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf


class Agent(object):

    def __init__(self, env, args):
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.args = args

        self.sess, device = self._make_session()
        with tf.device(device):
            self.construct_model()
        self.sess.run(tf.global_variables_initializer())

    def construct_model(self):
        with tf.variable_scope('graph_input'):
            self.ph_state = tf.placeholder(dtype=tf.float32,
                                           shape=[None, self.state_dim],
                                           name='input_state')
            self.ph_discounted_r = tf.placeholder(dtype=tf.float32,
                                                  shape=[None, 1],
                                                  name='discounted_reward')
            self.ph_action = tf.placeholder(dtype=tf.float32,
                                            shape=[None, self.action_dim],
                                            name='input_action')
            self.ph_lambda = tf.placeholder(dtype=tf.float32,
                                            shape=None,
                                            name='lambda')

        with tf.variable_scope('actor'):
            pi = self._build_actor(self.ph_state)
            pi_params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 'actor')
        with tf.variable_scope('old_actor'):
            old_pi = self._build_actor(self.ph_state, trainable=False)
            old_pi_params = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 'old_actor')

        with tf.variable_scope('critic'):
            self.state_value = self._build_critic(self.ph_state)

        with tf.variable_scope('sample_action'):
            self.sample_action_op = tf.squeeze(pi.sample(1), axis=0)

        with tf.variable_scope('build_train'):
            # critic loss
            self.advantage = self.ph_discounted_r - self.state_value
            critic_loss = tf.reduce_mean(tf.square(self.advantage))
            # actor loss
            ratio = pi.prob(self.ph_action) / old_pi.prob(self.ph_action)
            surrogate = ratio * tf.stop_gradient(self.advantage)
            if self.args.method == 'kl_pen':
                kl = tf.contrib.distributions.kl_divergence(old_pi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                actor_loss = -tf.reduce_mean(surrogate - self.ph_lambda * kl)
            elif self.args.method == 'clip':
                actor_obj = tf.minimum(
                    surrogate,
                    tf.stop_gradient(self.advantage) * tf.clip_by_value(
                        ratio,
                        1 - self.args.epsilon,
                        1 + self.args.epsilon))
                actor_loss = -tf.reduce_mean(actor_obj)
            else:
                raise NotImplementedError()
            # train op
            self.train_actor_op = tf.train.AdamOptimizer(
                self.args.actor_lr).minimize(actor_loss)
            self.train_critic_op = tf.train.AdamOptimizer(
                self.args.critic_lr).minimize(critic_loss)

        with tf.variable_scope('update_old_pi'):
            update_old_pi_op = []
            for pi, old_pi in zip(pi_params, old_pi_params):
                update_old_pi_op.append(old_pi.assign(pi))
                # update_op = old_pi.assign_sub(0.1 * (old_pi - pi))
                # update_old_pi_op.append(update_op)
            self.update_old_pi_op = tf.group(*update_old_pi_op)

    def sample_action(self, input_state):
        input_state = input_state[np.newaxis, :]
        a = self.sess.run(
            self.sample_action_op, {self.ph_state: input_state})[0]
        return np.clip(a, -2.0, 2.0)

    def get_value(self, input_state):
        if input_state.ndim < 2:
            input_state = input_state[np.newaxis, :]
        state_value = self.sess.run(
            self.state_value, {self.ph_state: input_state})[0][0]
        return state_value

    def update_model(self, b_s, b_a, b_r):
        self.sess.run(self.update_old_pi_op)
        # update actor
        if self.args.method == 'kl_pen':
            for _ in range(self.args.a_update_steps):
                feed_dict = {self.ph_state: b_s,
                             self.ph_action: b_a,
                             self.ph_discounted_r: b_r,
                             self.ph_lambda: self.args.lamb}
                _, kl = self.sess.run(
                    [self.train_actor_op, self.kl_mean], feed_dict)
                if kl > 4 * self.args.kl_target:
                    break
            if kl < self.args.kl_target / 1.5:
                self.args.lamb /= 2.0
            elif kl > self.args.kl_target * 1.5:
                self.args.lamb *= 2.0
            self.args.lamb = np.clip(self.args.lamb, 0.0001, 10)
        elif self.args.method == 'clip':
            for _ in range(self.args.a_update_steps):
                self.sess.run(
                    self.train_actor_op,
                    {self.ph_state: b_s,
                     self.ph_action: b_a,
                     self.ph_discounted_r: b_r})
        else:
            raise NotImplementedError()

        # update critic
        for _ in range(self.args.c_update_steps):
            self.sess.run(
                self.train_critic_op,
                {self.ph_state: b_s, self.ph_discounted_r: b_r})

    def _build_actor(self, input_state, trainable=True):
        hidden_unit = 50
        h1 = tf.layers.dense(
            input_state, hidden_unit, tf.nn.relu, trainable=trainable)
        mu = 2.0 * tf.layers.dense(
            h1, self.action_dim, tf.nn.tanh, trainable=trainable)
        sigma = tf.layers.dense(
            h1, self.action_dim, tf.nn.softplus, trainable=trainable)
        pi = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
        return pi

    def _build_critic(self, input_state, trainable=True):
        hidden_unit = 50
        h1 = tf.layers.dense(
            input_state, hidden_unit, tf.nn.relu, trainable=trainable)
        state_value = tf.layers.dense(h1, 1, trainable=trainable)
        return state_value

    def _make_session(self):
        if self.args.gpu == -1:  # use CPU
            device = '/cpu:0'
            sess_config = tf.ConfigProto()
        else:  # use GPU
            device = '/gpu:' + str(self.args.gpu)
            sess_config = tf.ConfigProto(
                log_device_placement=True, allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True

        return tf.Session(config=sess_config), device
