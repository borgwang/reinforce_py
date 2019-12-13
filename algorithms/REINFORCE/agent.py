import numpy as np
import tensorflow as tf


class REINFORCE:

    def __init__(self, input_dim, hidden_units, action_dim):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.action_dim = action_dim
        self.gamma = 0.99
        self.max_gradient = 5

        self.state_buffer = []
        self.reward_buffer = []
        self.action_buffer = []

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
            # construct network
            self.input_state = tf.placeholder(
                tf.float32, [None, self.input_dim])
            w1 = tf.Variable(tf.div(tf.random_normal(
                [self.input_dim, self.hidden_units]),
                np.sqrt(self.input_dim)))
            b1 = tf.Variable(tf.constant(0.0, shape=[self.hidden_units]))
            h1 = tf.nn.relu(tf.matmul(self.input_state, w1) + b1)
            w2 = tf.Variable(tf.div(
                tf.random_normal([self.hidden_units, self.action_dim]),
                np.sqrt(self.hidden_units)))
            b2 = tf.Variable(tf.constant(0.0, shape=[self.action_dim]))

            self.logp = tf.matmul(h1, w2) + b2

            self.discounted_rewards = tf.placeholder(tf.float32, [None, ])
            self.taken_actions = tf.placeholder(tf.int32, [None, ])

            # optimizer
            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=1e-4, decay=0.99)
            # loss
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logp, labels=self.taken_actions))
            # gradient
            self.gradient = self.optimizer.compute_gradients(self.loss)
            # policy gradient
            for i, (grad, var) in enumerate(self.gradient):
                if grad is not None:
                    pg_grad = grad * self.discounted_rewards
                    # gradient clipping
                    pg_grad = tf.clip_by_value(
                        pg_grad, -self.max_gradient, self.max_gradient)
                    self.gradient[i] = (pg_grad, var)
            # train operation (apply gradient)
            self.train_op = self.optimizer.apply_gradients(self.gradient)

    def sample_action(self, state):

        def softmax(x):
            max_x = np.amax(x)
            e = np.exp(x - max_x)
            return e / np.sum(e)

        logp = self.sess.run(self.logp, {self.input_state: state})[0]
        prob = softmax(logp) - 1e-5
        action = np.argmax(np.random.multinomial(1, prob))
        return action

    def update_model(self):
        discounted_rewards = self.reward_discount()
        episode_steps = len(discounted_rewards)

        for s in reversed(range(episode_steps)):
            state = self.state_buffer[s][np.newaxis, :]
            action = np.array([self.action_buffer[s]])
            reward = np.array([discounted_rewards[s]])
            self.sess.run(self.train_op, {
                self.input_state: state,
                self.taken_actions: action,
                self.discounted_rewards: reward
            })

        # cleanup job
        self.state_buffer = []
        self.reward_buffer = []
        self.action_buffer = []

    def store_rollout(self, state, action, reward):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)

    def reward_discount(self):
        r = self.reward_buffer
        d_r = np.zeros_like(r)
        running_add = 0
        for t in range(len(r))[::-1]:
            if r[t] != 0:
                running_add = 0  # game boundary. reset the running add
            running_add = r[t] + running_add * self.gamma
            d_r[t] += running_add
        # standardize the rewards
        d_r -= np.mean(d_r)
        d_r /= np.std(d_r)
        return d_r
