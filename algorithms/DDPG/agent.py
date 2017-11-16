from __future__ import print_function
from __future__ import division

import random
from collections import deque
import tensorflow as tf
import numpy as np

from ou_noise import OUNoise


class DDPG(object):

    def __init__(self, env):
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.h1_dim = 400
        self.h2_dim = 300

        self.actor_learning_rate = 1e-4
        self.critic_learning_rate = 1e-3

        self.gamma = 0.99

        # Ornstein-Uhlenbeck noise parameters
        self.noise_theta = 0.15
        self.noise_sigma = 0.20
        self.ou = OUNoise(
            self.action_dim, theta=self.noise_theta, sigma=self.noise_sigma)

        self.replay_buffer_size = 1000000
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)
        self.replay_start_size = 1000

        self.batch_size = 64

        self.target_update_rate = 0.001
        self.total_parameters = 0
        self.global_steps = 0
        self.reg_param = 0.01

    def construct_model(self, gpu):
        if gpu == -1:  # use CPU
            device = '/cpu:0'
            sess_config = tf.ConfigProto()
        else:  # use GPU
            device = '/gpu:' + str(gpu)
            sess_config = tf.ConfigProto(
                log_device_placement=True, allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config)

        with tf.device(device):
            # output action and q_value and compute gradients of q_val
            #  w.r.t. action
            with tf.name_scope('predict_actions'):
                self.states = tf.placeholder(
                    tf.float32, [None, self.state_dim], name='states')
                self.action = tf.placeholder(
                    tf.float32, [None, self.action_dim], name='action')
                self.is_training = tf.placeholder(tf.bool, name='is_training')

                with tf.variable_scope('actor_network'):
                    self.action_outputs, self.actor_params = \
                        self.actor(self.states, bn=True)
                with tf.variable_scope('critic_network'):
                    self.value_outputs, self.critic_params = \
                        self.critic(self.states, self.action, bn=False)
                    self.action_gradients = tf.gradients(
                        self.value_outputs, self.action)[0]

            # estimate target_q for update critic
            with tf.name_scope('estimate_target_q'):
                self.next_states = tf.placeholder(
                    tf.float32, [None, self.state_dim], name='next_states')
                self.mask = tf.placeholder(
                    tf.float32, [None, ], name='mask')
                self.rewards = tf.placeholder(
                    tf.float32, [None, ], name='rewards')

                with tf.variable_scope('target_actor_network'):
                    self.target_actor_outputs, self.target_actor_params = \
                        self.actor(self.next_states, bn=True)
                with tf.variable_scope('target_critic_network'):
                    self.target_value_outputs, self.target_critic_params = \
                        self.critic(self.next_states,
                                    self.target_actor_outputs,
                                    bn=False)

                self.target_q = self.rewards + self.gamma * \
                    (self.target_value_outputs[:, 0] * self.mask)

            with tf.name_scope('compute_gradients'):
                # optimizer
                self.actor_optimizer = tf.train.AdamOptimizer(
                    self.actor_learning_rate)
                self.critic_optimizer = tf.train.AdamOptimizer(
                    self.critic_learning_rate)
                # critic gradients
                td_error = self.target_q - self.value_outputs[:, 0]
                critic_mse = tf.reduce_mean(tf.square(td_error))
                critic_reg = tf.reduce_sum(
                    [tf.nn.l2_loss(v) for v in self.critic_params])
                critic_loss = critic_mse + self.reg_param * critic_reg
                self.critic_gradients = \
                    self.critic_optimizer.compute_gradients(
                        critic_loss, self.critic_params)
                # actor gradients
                self.q_action_grads = tf.placeholder(
                    tf.float32, [None, self.action_dim],
                    name='q_action_grads')
                actor_gradients = tf.gradients(
                    self.action_outputs, self.actor_params,
                    -self.q_action_grads)
                self.actor_gradients = zip(actor_gradients, self.actor_params)
                # apply gradient to update model
                self.train_actor = self.actor_optimizer.apply_gradients(
                    self.actor_gradients)
                self.train_critic = self.critic_optimizer.apply_gradients(
                    self.critic_gradients)

            with tf.name_scope('update_target_networks'):
                # batch norm paramerters should not be included when updating!
                target_networks_update = []

                for v_source, v_target in zip(
                        self.actor_params, self.target_actor_params):
                    update_op = v_target.assign_sub(
                        self.target_update_rate * (v_target - v_source))
                    target_networks_update.append(update_op)

                for v_source, v_target in zip(
                        self.critic_params, self.target_critic_params):
                    update_op = v_target.assign_sub(
                        self.target_update_rate * (v_target - v_source))
                    target_networks_update.append(update_op)

                self.target_networks_update = tf.group(*target_networks_update)

            with tf.name_scope('total_numbers_of_parameters'):
                for v in tf.trainable_variables():
                    shape = v.get_shape()
                    param_num = 1
                    for d in shape:
                        param_num *= d.value
                    print(v.name, ' ', shape, ' param nums: ', param_num)
                    self.total_parameters += param_num
                print('Total nums of parameters: ', self.total_parameters)

    def sample_action(self, states, explore):
        # is_training suppose to be False when sampling action!!!
        action = self.sess.run(self.action_outputs, {
            self.states: states,
            self.is_training: False
        })
        ou_noise = self.ou.noise() if explore else 0

        return action + ou_noise

    def store_experience(self, s, a, r, next_s, done):
        self.replay_buffer.append([s, a[0], r, next_s, done])
        self.global_steps += 1

    def update_model(self):

        if len(self.replay_buffer) < self.replay_start_size:
            return

        # get batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        s, _a, r, next_s, done = np.vstack(batch).T.tolist()
            mask = ~np.array(done)

        # compute a = u(s)
        a = self.sess.run(self.action_outputs, {
            self.states: s,
            self.is_training: True
        })
        # gradients of q_value w.r.t action a
        dq_da = self.sess.run(self.action_gradients, {
            self.states: s,
            self.action: a,
            self.is_training: True
        })
        # train
        self.sess.run([self.train_actor, self.train_critic], {
            # train_actor feed
            self.states: s,
            self.is_training: True,
            self.q_action_grads: dq_da,
            # train_critic feed
            self.next_states: next_s,
            self.action: _a,
            self.mask: mask,
            self.rewards: r
        })
        # update target network
        self.sess.run(self.target_networks_update)

    def actor(self, states, bn=False):
        init = tf.contrib.layers.variance_scaling_initializer(
            factor=1.0, mode='FAN_IN', uniform=True)
        if bn:
            states = self.batch_norm(
                states, self.is_training, tf.identity, scope='actor_bn_states')

        w1 = tf.get_variable(
            'w1', [self.state_dim, self.h1_dim], initializer=init)
        b1 = tf.get_variable('b1', [self.h1_dim], initializer=init)
        h1 = tf.matmul(states, w1) + b1
        if bn:
            h1 = self.batch_norm(
                h1, self.is_training, tf.nn.relu, scope='actor_bn_h1')

        w2 = tf.get_variable(
            'w2', [self.h1_dim, self.h2_dim], initializer=init)
        b2 = tf.get_variable('b2', [self.h2_dim], initializer=init)
        h2 = tf.matmul(h1, w2) + b2
        if bn:
            h2 = self.batch_norm(
                h2, self.is_training, tf.nn.relu, scope='actor_bn_h2')

        # use tanh to bound the action
        w3 = tf.get_variable(
            'w3', [self.h2_dim, self.action_dim],
            initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
        b3 = tf.get_variable(
            'b3', [self.action_dim],
            initializer=tf.random_uniform_initializer(-3e-4, 3e-4))
        a = tf.nn.tanh(tf.matmul(h2, w3) + b3)

        return a, [w1, b1, w2, b2, w3, b3]

    def critic(self, states, action, bn=False):
        init = tf.contrib.layers.variance_scaling_initializer(
            factor=1.0, mode='FAN_IN', uniform=True)
        if bn:
            states = self.batch_norm(
                states, self.is_training, tf.identity, scope='critic_bn_state')

        w1 = tf.get_variable(
            'w1', [self.state_dim, self.h1_dim], initializer=init)
        b1 = tf.get_variable('b1', [self.h1_dim], initializer=init)
        h1 = tf.matmul(states, w1) + b1
        if bn:
            h1 = self.batch_norm(
                h1, self.is_training, tf.nn.relu, scope='critic_bn_h1')

        # skip action from the first layer
        h1_concat = tf.concat([h1, action], 1)

        w2 = tf.get_variable(
            'w2', [self.h1_dim + self.action_dim, self.h2_dim], initializer=init)
        b2 = tf.get_variable('b2', [self.h2_dim], initializer=init)
        h2 = tf.nn.relu(tf.matmul(h1_concat, w2) + b2)

        w3 = tf.get_variable(
            'w3', [self.h2_dim, 1],
            initializer=tf.random_uniform_initializer(-3e-3, 3e-3))
        b3 = tf.get_variable(
            'b3', [1], initializer=tf.random_uniform_initializer(-3e-4, 3e-4))
        q = tf.matmul(h2, w3) + b3

        return q, [w1, b1, w2, b2, w3, b3]

    def batch_norm(self, x, is_training, activation_fn, scope):
        # switch the 'is_training' flag and 'reuse' flag
        return tf.cond(
            is_training,
            lambda: tf.contrib.layers.batch_norm(
                x,
                activation_fn=activation_fn,
                center=True,
                scale=True,
                updates_collections=None,
                is_training=True,
                reuse=None,
                scope=scope,
                decay=0.9,
                epsilon=1e-5),
            lambda: tf.contrib.layers.batch_norm(
                x,
                activation_fn=activation_fn,
                center=True,
                scale=True,
                updates_collections=None,
                is_training=False,
                reuse=True,  # to be able to reuse scope must be given
                scope=scope,
                decay=0.9,
                epsilon=1e-5))
