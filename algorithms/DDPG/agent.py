import random
from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl

from ou_noise import OUNoise


class DDPG:

    def __init__(self, env, args):
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]

        self.actor_lr = args.a_lr
        self.critic_lr = args.c_lr

        self.gamma = args.gamma

        # Ornstein-Uhlenbeck noise parameters
        self.ou = OUNoise(
            self.action_dim, theta=args.noise_theta, sigma=args.noise_sigma)

        self.replay_buffer = deque(maxlen=args.buffer_size)
        self.replay_start_size = args.replay_start_size

        self.batch_size = args.batch_size

        self.target_update_rate = args.target_update_rate
        self.total_parameters = 0
        self.global_steps = 0
        self.reg_param = args.reg_param

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
            # output action, q_value and gradients of q_val w.r.t. action
            with tf.name_scope('predict_actions'):
                self.states = tf.placeholder(
                    tf.float32, [None, self.state_dim], name='states')
                self.action = tf.placeholder(
                    tf.float32, [None, self.action_dim], name='action')
                self.is_training = tf.placeholder(tf.bool, name='is_training')

                self.action_outputs, actor_params = self._build_actor(
                    self.states, scope='actor_net', bn=True)
                value_outputs, critic_params = self._build_critic(
                    self.states, self.action, scope='critic_net', bn=False)
                self.action_gradients = tf.gradients(
                    value_outputs, self.action)[0]

            # estimate target_q for update critic
            with tf.name_scope('estimate_target_q'):
                self.next_states = tf.placeholder(
                    tf.float32, [None, self.state_dim], name='next_states')
                self.mask = tf.placeholder(tf.float32, [None], name='mask')
                self.rewards = tf.placeholder(tf.float32, [None], name='rewards')

                # target actor network
                t_action_outputs, t_actor_params = self._build_actor(
                    self.next_states, scope='t_actor_net', bn=True,
                    trainable=False)
                # target critic network
                t_value_outputs, t_critic_params = self._build_critic(
                    self.next_states, t_action_outputs, bn=False,
                    scope='t_critic_net', trainable=False)

                target_q = self.rewards + self.gamma * \
                    (t_value_outputs[:, 0] * self.mask)

            with tf.name_scope('compute_gradients'):
                actor_opt = tf.train.AdamOptimizer(self.actor_lr)
                critic_opt = tf.train.AdamOptimizer(self.critic_lr)

                # critic gradients
                td_error = target_q - value_outputs[:, 0]
                critic_mse = tf.reduce_mean(tf.square(td_error))
                critic_reg = tf.reduce_sum(
                    [tf.nn.l2_loss(v) for v in critic_params])
                critic_loss = critic_mse + self.reg_param * critic_reg
                critic_gradients = critic_opt.compute_gradients(
                    critic_loss, critic_params)
                # actor gradients
                self.q_action_grads = tf.placeholder(
                    tf.float32, [None, self.action_dim], name='q_action_grads')
                actor_gradients = tf.gradients(
                    self.action_outputs, actor_params, -self.q_action_grads)
                actor_gradients = zip(actor_gradients, actor_params)
                # apply gradient to update model
                self.train_actor = actor_opt.apply_gradients(actor_gradients)
                self.train_critic = critic_opt.apply_gradients(
                    critic_gradients)

            with tf.name_scope('update_target_networks'):
                # batch norm parameters should not be included when updating!
                target_networks_update = []

                for v_source, v_target in zip(actor_params, t_actor_params):
                    update_op = v_target.assign_sub(
                        0.001 * (v_target - v_source))
                    target_networks_update.append(update_op)

                for v_source, v_target in zip(critic_params, t_critic_params):
                    update_op = v_target.assign_sub(
                        0.01 * (v_target - v_source))
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

    def sample_action(self, states, noise):
        # is_training suppose to be False when sampling action.
        action = self.sess.run(
            self.action_outputs,
            feed_dict={self.states: states, self.is_training: False})
        ou_noise = self.ou.noise() if noise else 0

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

    def _build_actor(self, states, scope, bn=False, trainable=True):
        h1_dim = 400
        h2_dim = 300
        init = tf.contrib.layers.variance_scaling_initializer(
            factor=1.0, mode='FAN_IN', uniform=True)

        with tf.variable_scope(scope):
            if bn:
                states = self.batch_norm(
                    states, self.is_training, tf.identity,
                    scope='actor_bn_states', trainable=trainable)
            h1 = tcl.fully_connected(
                states, h1_dim, activation_fn=None, weights_initializer=init,
                biases_initializer=init, trainable=trainable, scope='actor_h1')

            if bn:
                h1 = self.batch_norm(
                    h1, self.is_training, tf.nn.relu, scope='actor_bn_h1',
                    trainable=trainable)
            else:
                h1 = tf.nn.relu(h1)

            h2 = tcl.fully_connected(
                h1, h2_dim, activation_fn=None, weights_initializer=init,
                biases_initializer=init, trainable=trainable, scope='actor_h2')
            if bn:
                h2 = self.batch_norm(
                    h2, self.is_training, tf.nn.relu, scope='actor_bn_h2',
                    trainable=trainable)
            else:
                h2 = tf.nn.relu(h2)

            # use tanh to bound the action
            a = tcl.fully_connected(
                h2, self.action_dim, activation_fn=tf.nn.tanh,
                weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                biases_initializer=tf.random_uniform_initializer(-3e-4, 3e-4),
                trainable=trainable, scope='actor_out')

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

        return a, params

    def _build_critic(self, states, action, scope, bn=False, trainable=True):
        h1_dim = 400
        h2_dim = 300
        init = tf.contrib.layers.variance_scaling_initializer(
            factor=1.0, mode='FAN_IN', uniform=True)
        with tf.variable_scope(scope):
            if bn:
                states = self.batch_norm(
                    states, self.is_training, tf.identity,
                    scope='critic_bn_state', trainable=trainable)
            h1 = tcl.fully_connected(
                states, h1_dim, activation_fn=None, weights_initializer=init,
                biases_initializer=init, trainable=trainable, scope='critic_h1')
            if bn:
                h1 = self.batch_norm(
                    h1, self.is_training, tf.nn.relu, scope='critic_bn_h1',
                    trainable=trainable)
            else:
                h1 = tf.nn.relu(h1)

            # skip action from the first layer
            h1 = tf.concat([h1, action], 1)

            h2 = tcl.fully_connected(
                h1, h2_dim, activation_fn=None, weights_initializer=init,
                biases_initializer=init, trainable=trainable,
                scope='critic_h2')

            if bn:
                h2 = self.batch_norm(
                    h2, self.is_training, tf.nn.relu, scope='critic_bn_h2',
                    trainable=trainable)
            else:
                h2 = tf.nn.relu(h2)

            q = tcl.fully_connected(
                h2, 1, activation_fn=None,
                weights_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                biases_initializer=tf.random_uniform_initializer(-3e-4, 3e-4),
                trainable=trainable, scope='critic_out')

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)
        return q, params

    @staticmethod
    def batch_norm(x, is_training, activation_fn, scope, trainable=True):
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
                epsilon=1e-5,
                trainable=trainable),
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
                epsilon=1e-5,
                trainable=trainable))
