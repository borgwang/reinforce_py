import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper parameters
class DQN():
    def __init__(self, env):
        # Init replay buffer
        self.replay_buffer = deque()
        self.memory_size = 1000
        # Init parameters
        self.global_step = 0
        self.epsilon = 0.5
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.gamma = 0.99
        self.decay_rate = 0.99
        self.learning_rate = 1e-4

        self.batch_size = 32

    def network(self, input_state):
        hidden_unit = 50
        w1 = tf.Variable(tf.div(tf.random_normal(
                [self.state_dim, hidden_unit]), np.sqrt(self.state_dim)))
        b1 = tf.Variable(tf.constant(0.0, shape=[hidden_unit]))
        hidden = tf.nn.relu(tf.matmul(input_state, w1) + b1)

        w2 = tf.Variable(tf.div(tf.random_normal(
                [hidden_unit, self.action_dim]), np.sqrt(hidden_unit)))
        b2 = tf.Variable(tf.constant(0.0, shape=[self.action_dim]))
        output_Q = tf.matmul(hidden, w2) + b2

        return output_Q

    def construct_model(self, gpu):
        if gpu == -1: # use CPU
            device = '/cpu:0'
            sess_config = tf.ConfigProto()
        else: # use GPU
            device = '/gpu:' + str(gpu)
            sess_config = tf.ConfigProto(
                            log_device_placement=True,
                            allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config)

        with tf.name_scope('input_state'):
            self.input_state = tf.placeholder(tf.float32, [None, self.state_dim])

        with tf.name_scope('q_network'):
            self.output_Q = self.network(self.input_state)

        with tf.name_scope('optimize'):
            self.input_action = tf.placeholder(tf.float32, [None, self.action_dim])
            self.target_Q = tf.placeholder(tf.float32, [None])
            # Q value of the selceted action
            action_Q = tf.reduce_sum(tf.mul(self.output_Q, self.input_action), reduction_indices=1)

            self.loss = tf.reduce_mean(tf.square(self.target_Q - action_Q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, self.decay_rate).minimize(self.loss)

        # Target network
        with tf.name_scope('target_network'):
            self.target_output_Q = self.network(self.input_state)

        q_parameters = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_network')
        target_q_parameters = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network')

        with tf.name_scope('update_target_network'):
            self.update_target_network = []
            for v_source, v_target in zip(q_parameters, target_q_parameters):
                update_op = v_target.assign(v_source)
                self.update_target_network.append(update_op)
            # group all update together
            self.update_target_network = tf.group(*self.update_target_network)

    def init_model(self):
        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def sample_action(self, state, policy):
        self.global_step += 1
        # Q_value of all actions
        output_Q = self.sess.run(self.output_Q, feed_dict={self.input_state:[state]})[0]
        if policy == "egreedy":
            if random.random() <= self.epsilon: # random action
                return random.randint(0, self.action_dim-1)
            else:   # greedy action
                return  np.argmax(output_Q)
        elif policy == "greedy":
            return np.argmax(output_Q)
        elif policy == "random":
            return random.randint(0, self.action_dim-1)

    def learn(self,state,action,reward,next_state,done):
        onehot_action = np.zeros(self.action_dim)
        onehot_action[action] = 1

        # Store experience in deque
        self.replay_buffer.append(np.array([state,onehot_action,reward,next_state,done]))
        if len(self.replay_buffer) > self.memory_size:
            self.replay_buffer.popleft()
        if len(self.replay_buffer) > self.batch_size:
            self.update_model()

    def update_model(self):
        time_cost = []
        # Update target network
        if self.global_step % 1000 == 0:
            self.sess.run(self.update_target_network)
        # Sample experience
        minibatch = random.sample(self.replay_buffer, self.batch_size)

        # Transpose minibatch
        minibatch_t = np.array(minibatch).T
        s_batch, a_batch, r_batch, next_s_batch, done_batch = minibatch_t.tolist()

        # next_state_Q_value_batch = self.sess.run(self.output_Q,
        #             feed_dict={self.input_state:next_state_batch})

        next_state_Q_value_batch = \
                self.sess.run(self.target_output_Q, {self.input_state:next_s_batch})

        # Calculate target_Q_batch
        target_Q_batch = []
        for i in xrange(self.batch_size):
            done_state = done_batch[i]
            if done_state:
                target_Q_batch.append(r_batch[i])
            else:
                target_Q_batch.append(r_batch[i] + self.gamma * np.max(next_state_Q_value_batch[i]))

        # Run the optimizer. Train the network
        self.sess.run(self.optimizer, {
                self.target_Q: target_Q_batch,
                self.input_action: a_batch,
                self.input_state: s_batch
        })
