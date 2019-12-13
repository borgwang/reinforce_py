import numpy as np
import tensorflow as tf


class ActorCritic:

    def __init__(self, input_dim, hidden_units, action_dim):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.action_dim = action_dim
        self.gamma = 0.99
        self.discount_factor = 0.99
        self.max_gradient = 5
        # counter
        self.ep_count = 0
        # buffer init
        self.buffer_reset()

        self.batch_size = 32

    @staticmethod
    def get_session(device):
        if device == -1:  # use CPU
            device = '/cpu:0'
            sess_config = tf.ConfigProto()
        else:  # use GPU
            device = '/gpu:' + str(device)
            sess_config = tf.ConfigProto(
                log_device_placement=True,
                allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        return sess, device

    def construct_model(self, gpu):
        self.sess, device = self.get_session(gpu)

        with tf.device(device):
            with tf.name_scope('model_inputs'):
                self.input_state = tf.placeholder(
                    tf.float32, [None, self.input_dim], name='input_state')
            with tf.variable_scope('actor_network'):
                self.logp = self.actor_network(self.input_state)
            with tf.variable_scope('critic_network'):
                self.state_value = self.critic_network(self.input_state)

            # get network parameters
            actor_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_network')
            critic_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_network')

            self.taken_action = tf.placeholder(tf.int32, [None, ])
            self.discounted_rewards = tf.placeholder(tf.float32, [None, 1])

            # optimizer
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4)
            # actor loss
            self.actor_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logp, labels=self.taken_action)
            # advantage
            self.advantage = (self.discounted_rewards - self.state_value)[:, 0]
            # actor gradient
            actor_gradients = tf.gradients(
                self.actor_loss, actor_params, self.advantage)
            self.actor_gradients = list(zip(actor_gradients, actor_params))

            # critic loss
            self.critic_loss = tf.reduce_mean(
                tf.square(self.discounted_rewards - self.state_value))
            # critic gradient
            self.critic_gradients = self.optimizer.compute_gradients(
                self.critic_loss, critic_params)
            self.gradients = self.actor_gradients + self.critic_gradients

            # clip gradient
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (tf.clip_by_value(
                            grad, -self.max_gradient, self.max_gradient), var)

            with tf.name_scope('train_actor_critic'):
                # train operation
                self.train_op = self.optimizer.apply_gradients(self.gradients)

    def sample_action(self, state):

        def softmax(x):
            max_x = np.amax(x)
            e = np.exp(x - max_x)
            return e / np.sum(e)

        logp = self.sess.run(self.logp, {self.input_state: state})[0]
        prob = softmax(logp) - 1e-5
        return np.argmax(np.random.multinomial(1, prob))

    def update_model(self):
        state_buffer = np.array(self.state_buffer)
        action_buffer = np.array(self.action_buffer)
        discounted_rewards_buffer = np.vstack(self.reward_discount())

        ep_steps = len(action_buffer)
        shuffle_index = np.arange(ep_steps)
        np.random.shuffle(shuffle_index)

        for i in range(0, ep_steps, self.batch_size):
            if self.batch_size <= ep_steps:
                end_index = i + self.batch_size
            else:
                end_index = ep_steps
            batch_index = shuffle_index[i:end_index]

            # get batch
            input_state = state_buffer[batch_index]
            taken_action = action_buffer[batch_index]
            discounted_rewards = discounted_rewards_buffer[batch_index]

            # train!
            self.sess.run(self.train_op, feed_dict={
                self.input_state: input_state,
                self.taken_action: taken_action,
                self.discounted_rewards: discounted_rewards})

        # clean up job
        self.buffer_reset()

        self.ep_count += 1

    def store_rollout(self, state, action, reward, next_state, done):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)
        self.next_state_buffer.append(next_state)
        self.done_buffer.append(done)

    def buffer_reset(self):
        self.state_buffer = []
        self.reward_buffer = []
        self.action_buffer = []
        self.next_state_buffer = []
        self.done_buffer = []

    def reward_discount(self):
        r = self.reward_buffer
        d_r = np.zeros_like(r)
        running_add = 0
        for t in range(len(r))[::-1]:
            if r[t] != 0:
                running_add = 0  # game boundary. reset the running add
            running_add = r[t] + running_add * self.discount_factor
            d_r[t] += running_add
        # standardize the rewards
        d_r -= np.mean(d_r)
        d_r /= np.std(d_r)
        return d_r

    def actor_network(self, input_state):
        w1 = tf.Variable(tf.div(tf.random_normal(
            [self.input_dim, self.hidden_units]),
            np.sqrt(self.input_dim)), name='w1')
        b1 = tf.Variable(
            tf.constant(0.0, shape=[self.hidden_units]), name='b1')
        h1 = tf.nn.relu(tf.matmul(input_state, w1) + b1)
        w2 = tf.Variable(tf.div(tf.random_normal(
            [self.hidden_units, self.action_dim]),
            np.sqrt(self.hidden_units)), name='w2')
        b2 = tf.Variable(tf.constant(0.0, shape=[self.action_dim]), name='b2')
        logp = tf.matmul(h1, w2) + b2
        return logp

    def critic_network(self, input_state):
        w1 = tf.Variable(tf.div(tf.random_normal(
            [self.input_dim, self.hidden_units]),
            np.sqrt(self.input_dim)), name='w1')
        b1 = tf.Variable(
            tf.constant(0.0, shape=[self.hidden_units]), name='b1')
        h1 = tf.nn.relu(tf.matmul(input_state, w1) + b1)
        w2 = tf.Variable(tf.div(tf.random_normal(
            [self.hidden_units, 1]), np.sqrt(self.hidden_units)), name='w2')
        b2 = tf.Variable(tf.constant(0.0, shape=[1]), name='b2')
        state_value = tf.matmul(h1, w2) + b2
        return state_value
