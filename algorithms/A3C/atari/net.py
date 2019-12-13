import tensorflow.contrib.slim as slim

from utils import *


class Net(object):
    '''
    An Actor-Critic Network class. The shallow layers are shared by the Actor
    and the Critic.

    Args:
        s_dim: dimensions of the state space
        a_dim: dimensions of the action space
        scope: Scope the net belongs to
        trainer: optimizer used by this net
    '''

    def __init__(self, s_dim, a_dim, scope, args, trainer=None):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.scope = scope
        self.smooth = args.smooth
        self.clip_grads = args.clip_grads
        self.entropy_ratio = args.entropy_ratio

        with tf.variable_scope(self.scope):
            self.inputs = tf.placeholder(tf.float32, shape=[None] + self.s_dim)

            self._contruct_network(self.inputs)

            if self.scope != 'global':
                self._update_network(trainer)

    def _contruct_network(self, inputs):
        '''
        Biuld the computational graph.
        '''
        conv1 = slim.conv2d(inputs=inputs,
                            num_outputs=32,
                            kernel_size=[8, 8],
                            stride=[4, 4],
                            padding='VALID',
                            weights_initializer=ortho_init(),
                            scope='share_conv1')
        conv2 = slim.conv2d(inputs=conv1,
                            num_outputs=64,
                            kernel_size=[4, 4],
                            stride=[2, 2],
                            padding='VALID',
                            weights_initializer=ortho_init(),
                            scope='share_conv2')
        conv3 = slim.conv2d(inputs=conv2,
                            num_outputs=64,
                            kernel_size=[3, 3],
                            stride=[1, 1],
                            padding='VALID',
                            weights_initializer=ortho_init(),
                            scope='share_conv3')
        fc = slim.fully_connected(inputs=slim.flatten(conv3),
                                  num_outputs=512,
                                  weights_initializer=ortho_init(np.sqrt(2)),
                                  scope='share_fc1')
        self.policy = slim.fully_connected(inputs=fc,
                                           num_outputs=self.a_dim,
                                           activation_fn=tf.nn.softmax,
                                           scope='policy_out')
        self.value = slim.fully_connected(inputs=fc, num_outputs=1,
                                          activation_fn=None,
                                          scope='value_out')

    def _update_network(self, trainer):
        '''
        Build losses, compute gradients and apply gradients to the global net
        '''

        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        actions_onehot = tf.one_hot(self.actions, self.a_dim, dtype=tf.float32)
        self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

        action_prob = tf.reduce_sum(self.policy * actions_onehot, [1])

        # MSE critic loss
        self.critic_loss = 0.5 * tf.reduce_sum(
                tf.squared_difference(
                    self.target_v, tf.reshape(self.value, [-1])))

        # high entropy -> low loss -> encourage exploration
        self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy + 1e-30), 1)
        self.entropy_loss = -self.entropy_ratio * tf.reduce_sum(self.entropy)

        # policy gradients = d_[-log(p) * advantages] / d_theta
        self.actor_loss = -tf.reduce_sum(
            tf.log(action_prob + 1e-30) * self.advantages)
        self.actor_loss += self.entropy_loss

        self.loss = self.actor_loss + self.critic_loss
        local_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        self.grads = tf.gradients(self.loss, local_vars)

        # global norm gradients clipping
        self.grads, self.grad_norms = \
            tf.clip_by_global_norm(self.grads, self.clip_grads)
        self.var_norms = tf.global_norm(local_vars)
        global_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.apply_grads_to_global = \
            trainer.apply_gradients(zip(self.grads, global_vars))

        # summaries
        if self.scope == 'worker_1':
            tf.summary.scalar('loss/entropy', tf.reduce_sum(self.entropy))
            tf.summary.scalar('loss/actor_loss', self.actor_loss)
            tf.summary.scalar('loss/critic_loss', self.critic_loss)
            tf.summary.scalar('advantages', tf.reduce_mean(self.advantages))
            tf.summary.scalar('norms/grad_norms', self.grad_norms)
            tf.summary.scalar('norms/var_norms', self.var_norms)
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            self.summaries = tf.summary.merge(summaries)
        else:
            self.summaries = tf.no_op()
