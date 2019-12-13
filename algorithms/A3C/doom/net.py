import tensorflow.contrib.slim as slim

from utils import *


class Net:

    def __init__(self, s_dim, a_dim, scope, trainer):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.scope = scope

        with tf.variable_scope(self.scope):
            self.inputs = tf.placeholder(
                shape=[None, self.s_dim], dtype=tf.float32)
            inputs = tf.reshape(self.inputs, [-1, 84, 84, 1])

            self._construct_network(inputs)

            if self.scope != 'global':
                # gradients update only for workers
                self._update_network(trainer)

    def _construct_network(self, inputs):
        # Actor network and critic network share all shallow layers
        conv1 = slim.conv2d(inputs=inputs,
                            num_outputs=16,
                            activation_fn=tf.nn.relu,
                            kernel_size=[8, 8],
                            stride=[4, 4],
                            padding='VALID')
        conv2 = slim.conv2d(inputs=conv1,
                            num_outputs=32,
                            activation_fn=tf.nn.relu,
                            kernel_size=[4, 4],
                            stride=[2, 2],
                            padding='VALID')
        hidden = slim.fully_connected(inputs=slim.flatten(conv2),
                                      num_outputs=256,
                                      activation_fn=tf.nn.relu)

        # Recurrent network for temporal dependencies
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=256)
        c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
        h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
        self.state_in = (c_in, h_in)

        rnn_in = tf.expand_dims(hidden, [0])
        step_size = tf.shape(inputs)[:1]
        state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

        lstm_out, lstm_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                                 inputs=rnn_in,
                                                 initial_state=state_in,
                                                 sequence_length=step_size,
                                                 time_major=False)
        lstm_c, lstm_h = lstm_state
        self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
        rnn_out = tf.reshape(lstm_out, [-1, 256])

        # output for policy and value estimations
        self.policy = slim.fully_connected(
            inputs=rnn_out,
            num_outputs=self.a_dim,
            activation_fn=tf.nn.softmax,
            weights_initializer=normalized_columns_initializer(0.01),
            biases_initializer=None)
        self.value = slim.fully_connected(
            inputs=rnn_out,
            num_outputs=1,
            activation_fn=None,
            weights_initializer=normalized_columns_initializer(1.0),
            biases_initializer=None)

    def _update_network(self, trainer):
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(
            self.actions, self.a_dim, dtype=tf.float32)
        self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

        self.outputs = tf.reduce_sum(
                self.policy * self.actions_onehot, [1])

        # loss
        self.value_loss = 0.5 * tf.reduce_sum(tf.square(
                self.target_v - tf.reshape(self.value, [-1])))
        # higher entropy -> lower loss -> encourage exploration
        self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy))

        self.policy_loss = -tf.reduce_sum(
            tf.log(self.outputs) * self.advantages)

        self.loss = 0.5 * self.value_loss \
            + self.policy_loss - 0.01 * self.entropy

        # local gradients
        local_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.gradients = tf.gradients(self.loss, local_vars)
        self.var_norms = tf.global_norm(local_vars)

        # grads[i] * clip_norm / max(global_norm, clip_norm)
        grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

        # apply gradients to global network
        global_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))
