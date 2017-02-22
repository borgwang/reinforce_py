import numpy as np
import tensorflow as tf

class ActorCritic(object):
    def __init__(self, input_dim, hidden_units, action_dim):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.action_dim = action_dim
        self.gamma = 0.99
        self.discount_factor = 0.99
        self.max_gradient = 5
        # buffer init
        self.buffer_reset()

    def construct_model(self, gpu):
        if gpu == -1: # use CPU
            device = '/cpu:0'
            sess_config = tf.ConfigProto(log_device_placement=True)
        else: # use GPU
            device = '/gpu:' + str(gpu)
            sess_config = tf.ConfigProto(log_device_placement=True,
                            allow_soft_placement = True)

        self.sess = tf.Session(config=sess_config)

        with tf.device(device):
            with tf.name_scope('model_inputs'):
                self.input_state = tf.placeholder(tf.float32, [None, self.input_dim])

            with tf.variable_scope('actor_network'):
                w1 = tf.Variable(tf.div(tf.random_normal(
                    [self.input_dim, self.hidden_units]), np.sqrt(self.input_dim)))
                b1 = tf.Variable(tf.constant(0.0, shape=[self.hidden_units]))
                h1 = tf.nn.relu(tf.matmul(self.input_state, w1) + b1)
                w2 = tf.Variable(tf.div(tf.random_normal(
                    [self.hidden_units, self.action_dim]), np.sqrt(self.hidden_units)))
                b2 = tf.Variable(tf.constant(0.0, shape=[self.action_dim]))
                self.logp = tf.matmul(h1, w2) + b2

            with tf.variable_scope('critic_network'):
                w1 = tf.Variable(tf.div(tf.random_normal(
                    [self.input_dim, self.hidden_units]), np.sqrt(self.input_dim)))
                b1 = tf.Variable(tf.constant(0.0, shape=[self.hidden_units]))
                h1 = tf.nn.relu(tf.matmul(self.input_state, w1) + b1)
                w2 = tf.Variable(tf.div(tf.random_normal(
                    [self.hidden_units, 1]), np.sqrt(self.hidden_units)))
                b2 = tf.Variable(tf.constant(0.0, shape=[1]))
                self.state_value = tf.matmul(h1, w2) + b2

            # get network parameters
            actor_parameters = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_network')
            critic_parameters = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_network')

            with tf.name_scope('actor_gradients'):
                self.discounted_rewards = tf.placeholder(tf.float32, [None,])
                self.taken_action = tf.placeholder(tf.int32, [None,])

                # optimizer
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4, decay=0.99)
                # actor loss
                self.actor_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(self.logp, self.taken_action))
                # actor gradient
                self.actor_gradients = self.optimizer.compute_gradients(self.actor_loss, actor_parameters)

                # advantage A(s) = (r + gamma * V(s')) - V(s)
                self.advantages = tf.reduce_sum(self.discounted_rewards - self.state_value)
                # policy gradient
                for i, (grad, var) in enumerate(self.actor_gradients):
                    if grad is not None:
                        pg_grad = grad * self.advantages
                        # gradient clipping
                        pg_grad = tf.clip_by_value(pg_grad, -self.max_gradient, self.max_gradient)
                        self.actor_gradients[i] = (pg_grad, var)

            with tf.name_scope('critic_gradients'):
                # critic loss
                self.critic_loss = tf.reduce_mean(tf.square(self.discounted_rewards - self.state_value))
                # critic gradient
                self.critic_gradients = self.optimizer.compute_gradients(self.critic_loss, critic_parameters)
                # clip gradient
                for i, (grad, var) in enumerate(self.critic_gradients):
                    if grad is not None:
                        self.critic_gradients[i] = (tf.clip_by_value(grad, -self.max_gradient, self.max_gradient), var)

            with tf.name_scope('train_actor_critic'):
                # train operation
                self.train_actor = self.optimizer.apply_gradients(self.actor_gradients)
                self.train_critic = self.optimizer.apply_gradients(self.critic_gradients)

    def init_var(self):
        # initialize variables
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)

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
        critic_loss = 0
        discounted_rewards = self.reward_discount()
        episode_steps = len(discounted_rewards)

        for s in reversed(xrange(episode_steps)):
            state = self.state_buffer[s][np.newaxis, :]
            action = np.array([self.action_buffer[s]])
            reward = np.array([discounted_rewards[s]])
            # train!
            self.sess.run([self.train_actor, self.train_critic], {
                self.input_state: state,
                self.taken_action: action,
                self.discounted_rewards: reward
            })
        # cleanup job
        self.buffer_reset()

    def store_rollout(self, state, action, reward):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)

    def buffer_reset(self):
        self.state_buffer  = []
        self.reward_buffer = []
        self.action_buffer = []

    def reward_discount(self):
        r = self.reward_buffer
        d_r = np.zeros_like(r)
        running_add = 0
        for t in range(len(r))[::-1]:
            if r[t] != 0:
                running_add = 0 # game boundary. reset the running add
            running_add = r[t] + running_add * self.discount_factor
            d_r[t] += running_add
        # standardize the rewards
        d_r -= np.mean(d_r)
        d_r /= np.std(d_r)
        return d_r
