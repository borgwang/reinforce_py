import random
from collections import deque
import tensorflow as tf
import numpy as np

class DDPG(object):
    def __init__(self, actor_network, critic_network, state_dim, action_dim):
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.max_gradient = 5

        self.discount_factor = 0.99

        # Ornstein-Uhlenbeck noise parameters
        self.noise_theta = 0.15
        self.noise_sigma = 0.20

        self.replay_buffer_size = 1000000
        self.replay_buffer = deque(maxlen=self.replay_buffer_size)

        self.batch_size = 32

        self.target_update_rate = 0.01

    def construct_model(self, gpu):
        if gpu == -1: # use CPU
            device = '/cpu:0'
            sess_config = tf.ConfigProto()
        else: # use GPU
            device = '/gpu:' + str(gpu)
            sess_config = tf.ConfigProto(log_device_placement=True,
                            allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)


        # model
        with tf.name_scope('model_inputs'):
            self.states = tf.placeholder(
                    tf.float32, [None, self.state_dim], name='states')
            self.action = tf.placeholder(
                    tf.float32, [None, self.action_dim], name='action')

        with tf.name_scope('predict_actions'):
            with tf.variable_scope('actor_network'):
                self.action_outputs = self.actor_network(self.states)
            with tf.variable_scope('critic_network'):
                self.value_outputs = self.critic_network(self.states, self.action)
                self.action_gradients = tf.gradients(self.value_outputs, self.action)[0]

        actor_parameters  = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_network')
        critic_parameters = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_network')
        with tf.name_scope('estimate_target_q'):
            self.next_states = tf.placeholder(
                    tf.float32, [None, self.state_dim], name='next_states')
            self.mask = tf.placeholder(
                    tf.float32, [None,], name='mask')
            self.rewards = tf.placeholder(tf.float32, [None,], name='reward')

            with tf.variable_scope('target_actor_network'):
                # sample action a' using target actor net in state s'
                self.target_actor_outputs = self.actor_network(self.next_states)
            with tf.variable_scope('target_critic_network'):
                self.target_critic_outputs = self.critic_network(self.next_states, self.target_actor_outputs)

            next_action_scores = tf.stop_gradient(self.target_critic_outputs)[:,0] * self.mask
            self.target_q = self.rewards + self.discount_factor * next_action_scores

        with tf.name_scope('compute_gradients'):
            # critic gradients
            self.td_error = self.target_q - self.value_outputs[:, 0]
            self.critic_loss = tf.reduce_mean(tf.square(self.td_error))
            self.critic_gradients = self.optimizer.compute_gradients(
                                            self.critic_loss, critic_parameters)
            # actor gradients
            self.q_action_grads = tf.placeholder(
                    tf.float32, [None, self.action_dim], name='q_action_grads')
            actor_gradients = tf.gradients(
                    self.action_outputs, actor_parameters, -self.q_action_grads)
            self.actor_gradients = zip(actor_gradients, actor_parameters)

            # collect all gradients
            self.gradients = self.actor_gradients + self.critic_gradients

            # clip gradients
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (tf.clip_by_value(grad, -self.max_gradient, self.max_gradient), var)

            # apply gradient to update model
            self.train_op = self.optimizer.apply_gradients(self.gradients)

        with tf.name_scope('compute_ou_noise'):
            self.noise_var = tf.Variable(tf.zeros([1, self.action_dim]))
            noise_random = tf.random_normal([1, self.action_dim], stddev=self.noise_sigma)
            self.ou_noise = self.noise_var.assign_sub(self.noise_theta * self.noise_var - noise_random)

        with tf.name_scope('update_target_network'):
            self.target_network_update = []
            # get parameters
            actor_parameters = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_network')
            target_actor_parameters = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor_network')
            for v_source, v_target in zip(actor_parameters, target_actor_parameters):
                # this is equivalent to target = (1-alpha) * target + alpha * source
                update_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
                self.target_network_update.append(update_op)
            # same for the critic network
            critic_parameters = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_network')
            target_critic_parameters = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic_network')
            for v_source, v_target in zip(critic_parameters, target_critic_parameters):
                # this is equivalent to target = (1-alpha) * target + alpha * source
                update_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
                self.target_network_update.append(update_op)

            # group all assignment operations together
            self.target_network_update = tf.group(*self.target_network_update)

    def init_model(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def sample_action(self, states, explore=True):
        action, ou_noise = self.sess.run([
            self.action_outputs,
            self.ou_noise
        ], {
            self.states: states
        })
        # add noise for exploration
        action = action + ou_noise if explore else action

        return action

    def store_experience(self, s, a, r, next_s, done):
        self.replay_buffer.append([s, a[0], r, next_s, done])

    def update_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        s, _a, r, next_s, done = np.vstack(batch).T.tolist()
        mask = -np.array(done)

        # compute a = u(s)
        a = self.sess.run(self.action_outputs, {
            self.states: s
        })
        # compute gradients of Q(s,a) w.r.t action a
        q_action_grads = self.sess.run(self.action_gradients, {
            self.states: s,
            self.action: a
        })
        self.sess.run(self.train_op, {
            self.states: s,
            self.next_states: next_s,
            self.action: _a,
            self.mask: mask,
            self.rewards: r,
            self.q_action_grads: q_action_grads
        })

        self.sess.run(self.target_network_update)
