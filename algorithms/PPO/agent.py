import numpy as np
import tensorflow as tf

from config import args, tf_config
from distributions import make_pd_type
import utils as U


class Policy:

    def __init__(self, ob_space, ac_space, batch, n_steps, reuse):
        ob_dim = (batch,) + ob_space.shape
        act_dim = ac_space.shape[0]
        self.ph_obs = tf.placeholder(tf.float32, ob_dim, name='ph_obs')

        with tf.variable_scope('policy', reuse=reuse):
            h1 = U.fc(self.ph_obs, out_dim=64, activation_fn=tf.nn.tanh,
                    init_scale=np.sqrt(2), scope='pi_fc1')
            h2 = U.fc(h1, out_dim=64, activation_fn=tf.nn.tanh,
                    init_scale=np.sqrt(2), scope='pi_fc2')
            pi = U.fc(h2, out_dim=act_dim, activation_fn=None, init_scale=0.01,
                    scope='pi')
            h1 = U.fc(self.ph_obs, out_dim=64, activation_fn=tf.nn.tanh,
                    init_scale=np.sqrt(2), scope='vf_fc1')
            h2 = U.fc(h1, out_dim=64, activation_fn=tf.nn.tanh,
                    init_scale=np.sqrt(2), scope='vf_fc2')
            vf = U.fc(h2, out_dim=1, activation_fn=None, scope='vf')[:, 0]
            logstd = tf.get_variable(name='logstd', shape=[1, act_dim],
                                     initializer=tf.zeros_initializer())
            # concatenate probabilities and logstds
            pd_params = tf.concat([pi, pi * 0.0 + logstd], axis=1)
            self.pd_type = make_pd_type(ac_space)
            self.pd = self.pd_type.pdfromflat(pd_params)

            self.a_out = self.pd.sample()
            self.neglogp = self.pd.get_neglogp(self.a_out)

            self.v_out = vf
            self.pi = pi


class PPO:

    def __init__(self, env):
        self.sess = tf.Session(config=tf_config)
        ob_space = env.observation_space
        ac_space = env.action_space
        self.act_policy = Policy(ob_space, ac_space, env.num_envs,
                                 n_steps=1, reuse=False)
        self.train_policy = Policy(ob_space, ac_space, args.minibatch,
                                   n_steps=args.batch_steps, reuse=True)
        self._build_train()
        self.sess.run(tf.global_variables_initializer())

    def _build_train(self):
        # build placeholders
        self.ph_obs_train = self.train_policy.ph_obs
        self.ph_a = self.train_policy.pd_type.get_action_placeholder([None])
        self.ph_adv = tf.placeholder(tf.float32, [None])
        self.ph_r = tf.placeholder(tf.float32, [None])
        self.ph_old_neglogp = tf.placeholder(tf.float32, [None])
        self.ph_old_v = tf.placeholder(tf.float32, [None])
        self.ph_lr = tf.placeholder(tf.float32, [])
        self.ph_clip_range = tf.placeholder(tf.float32, [])

        # build losses
        self.neglogp = self.train_policy.pd.get_neglogp(self.ph_a)
        self.entropy = tf.reduce_mean(self.train_policy.pd.get_entropy())
        v = self.train_policy.v_out
        v_clipped = self.ph_old_v + tf.clip_by_value(
            v - self.ph_old_v, -self.ph_clip_range, self.ph_clip_range)
        v_loss1 = tf.square(v - self.ph_r)
        v_loss2 = tf.square(v_clipped - self.ph_r)
        self.v_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss1, v_loss2))

        # ratio = tf.exp(self.ph_old_neglogp - self.neglogp)
        old_p = tf.exp(-self.ph_old_neglogp)
        new_p = tf.exp(-self.neglogp)
        ratio = new_p / old_p
        pg_loss1 = -self.ph_adv * ratio
        pg_loss2 = -self.ph_adv * tf.clip_by_value(
            ratio, 1.0 - self.ph_clip_range, 1.0 + self.ph_clip_range)
        self.pg_loss = tf.reduce_mean(tf.maximum(pg_loss1, pg_loss2))
        loss = self.pg_loss + args.v_coef * self.v_loss - \
            args.entropy_coef * self.entropy

        # build train operation
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'policy')
        grads = tf.gradients(loss, params)
        if args.max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(
                grads, args.max_grad_norm)
        grads = list(zip(grads, params))
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.ph_lr, epsilon=1e-5).apply_gradients(grads)

        # train info
        self.approxkl = 0.5 * tf.reduce_mean(
            tf.square(self.neglogp - self.ph_old_neglogp))
        self.clip_frac = tf.reduce_mean(
            tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.ph_clip_range)))
        self.avg_ratio = tf.reduce_mean(ratio)

    def step(self, obs, *_args, **_kwargs):
        feed_dict = {self.act_policy.ph_obs: obs}
        a, v, neglogp = self.sess.run(
            [self.act_policy.a_out,
             self.act_policy.v_out,
             self.act_policy.neglogp],
            feed_dict=feed_dict)
        return a, v, neglogp

    def get_value(self, obs, *_args, **_kwargs):
        feed_dict = {self.act_policy.ph_obs: obs}
        return self.sess.run(self.act_policy.v_out, feed_dict=feed_dict)

    def train(self, lr, clip_range, obs, returns, masks, actions, values, neglogps,
              advs):
        # advs = returns - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        feed_dict = {self.ph_obs_train: obs, self.ph_a: actions,
                     self.ph_adv: advs, self.ph_r: returns,
                     self.ph_old_neglogp: neglogps, self.ph_old_v: values,
                     self.ph_lr: lr,
                     self.ph_clip_range: clip_range}
        self.loss_names = ['loss_policy', 'loss_value', 'avg_ratio', 'policy_entropy',
                           'approxkl', 'clipfrac']
        return self.sess.run(
            [self.pg_loss, self.v_loss, self.avg_ratio, self.entropy,
             self.approxkl, self.clip_frac, self.train_op],
            feed_dict=feed_dict)[:-1]

