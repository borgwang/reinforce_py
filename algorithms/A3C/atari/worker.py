from atari_env import A_DIM
from atari_env import S_DIM
from net import Net
from utils import *


class Worker(object):
    '''
    An A3C worker thread. Run a game locally, gather gradients and apply
    to the global networks.

    Args:
        worker_id: A unique id for this thread
        env: Game environment used by this worker
        global_steps: Iterator that holds the global steps
        args: Global parameters and hyperparameters
    '''

    def __init__(
            self, worker_id, env, global_steps_counter, summary_writer, args):
        self.name = 'worker_' + str(worker_id)
        self.env = env
        self.args = args
        self.local_steps = 0
        self.global_steps_counter = global_steps_counter
        # each worker has its own optimizer and learning_rate
        self.learning_rate = tf.Variable(args.init_learning_rate,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name=self.name + '_lr')
        self.delta_lr = \
            args.init_learning_rate / (args.max_steps / args.threads)
        self.trainer = tf.train.RMSPropOptimizer(self.learning_rate,
                                                 decay=args.decay,
                                                 epsilon=args.epsilon)
        self.summary_writer = summary_writer

        self.local_net = Net(S_DIM,
                             A_DIM,
                             scope=self.name,
                             args=self.args,
                             trainer=self.trainer)

        self.update_local_op = self._update_local_vars()
        self.anneal_learning_rate = self._anneal_learning_rate()

    def run(self, sess, coord, saver):
        print('Starting %s...\n' % self.name)
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_op)
                rollout = []
                s = self.env.reset()
                while True:
                    p, v = sess.run(
                        [self.local_net.policy, self.local_net.value],
                        feed_dict={self.local_net.inputs: [s]})
                    a = np.random.choice(range(A_DIM), p=p[0])
                    s1, r, dead, done = self.env.step(a)
                    rollout.append([s, a, r, s1, dead, v[0][0]])
                    s = s1

                    global_steps = next(self.global_steps_counter)
                    self.local_steps += 1
                    sess.run(self.anneal_learning_rate)

                    if not dead and len(rollout) == self.args.tmax:
                        # calculate value of next state, uses for bootstraping
                        v1 = sess.run(self.local_net.value,
                                      feed_dict={self.local_net.inputs: [s]})
                        self._train(rollout, sess, v1[0][0], global_steps)
                        rollout = []
                        sess.run(self.update_local_op)

                    if dead:
                        break

                if len(rollout) != 0:
                    self._train(rollout, sess, 0.0, global_steps)
                # end condition
                if global_steps >= self.args.max_steps:
                    coord.request_stop()
                    print_time_cost(self.args.start_time)

    def _train(self, rollout, sess, bootstrap_value, global_steps):
        '''
        Update global networks based on the rollout experiences

        Args:
            rollout: A list of transitions experiences
            sess: Tensorflow session
            bootstrap_value: if the episode was not done, we bootstrap the value
                             from the last state.
            global_steps: use for summary
        '''

        rollout = np.array(rollout)
        observs, actions, rewards, next_observs, dones, values = rollout.T
        # compute advantages and discounted rewards
        rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = reward_discount(rewards_plus, self.args.gamma)[:-1]

        advantages = discounted_rewards - values

        summaries, _ = sess.run([
            self.local_net.summaries,
            self.local_net.apply_grads_to_global
        ], feed_dict={
            self.local_net.inputs: np.stack(observs),
            self.local_net.actions: actions,
            self.local_net.target_v: discounted_rewards,  # for value loss
            self.local_net.advantages: advantages  # for policy net
        })
        # write summaries
        if self.summary_writer and summaries:
            self.summary_writer.add_summary(summaries, global_steps)
            self.summary_writer.flush()

    def _update_local_vars(self):
        '''
        Assign global networks parameters to local networks
        '''
        global_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        local_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        update_op = []
        for g_v, l_v in zip(global_vars, local_vars):
            update_op.append(l_v.assign(g_v))

        return update_op

    def _anneal_learning_rate(self):
        return tf.cond(
            self.learning_rate > 0.0,
            lambda: tf.assign_sub(self.learning_rate, self.delta_lr),
            lambda: tf.assign(self.learning_rate, 0.0))
