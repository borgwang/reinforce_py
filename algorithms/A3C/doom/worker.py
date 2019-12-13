from net import Net
from utils import *
from vizdoom import *


class Worker(object):

    def __init__(self, worker_id, env, global_ep, args):
        self.name = 'worker_' + str(worker_id)
        self.env = env
        self.global_ep = global_ep
        self.args = args
        self.learning_rate = 1e-4
        self.gamma = 0.99
        self.trainer = tf.train.AdamOptimizer(self.learning_rate)

        # create local copy of AC network
        self.local_net = Net(self.env.state_dim,
                             self.env.action_dim,
                             scope=self.name,
                             trainer=self.trainer)

        self.update_local_op = self._update_local_params()

    def run(self, sess, coord, saver):
        running_reward = None
        ep_count = sess.run(self.global_ep)
        print('Starting ' + self.name)

        with sess.as_default(), sess.graph.as_default():

            while not coord.should_stop():
                sess.run(self.update_local_op)
                rollout = []
                ep_reward = 0
                ep_step_count = 0

                s = self.env.reset()
                self.ep_frames = []
                self.ep_frames.append(s)
                s = preprocess(s)
                rnn_state = self.local_net.state_init

                while True:
                    p, v, rnn_state = sess.run([
                        self.local_net.policy,
                        self.local_net.value,
                        self.local_net.state_out
                    ], {
                        self.local_net.inputs: [s],
                        self.local_net.state_in[0]: rnn_state[0],
                        self.local_net.state_in[1]: rnn_state[1]
                    })
                    # sample action from the policy distribution p
                    a = np.random.choice(np.arange(self.env.action_dim), p=p[0])

                    s1, r, d = self.env.step(a)
                    self.ep_frames.append(s1)
                    r /= 100.0  # scale rewards
                    s1 = preprocess(s1)

                    rollout.append([s, a, r, s1, d, v[0][0]])
                    ep_reward += r
                    s = s1
                    ep_step_count += 1

                    # Update if the buffer is full (size=30)
                    if not d and len(rollout) == 30 \
                             and ep_step_count != self.args.max_ep_len - 1:
                        v1 = sess.run(self.local_net.value, {
                            self.local_net.inputs: [s],
                            self.local_net.state_in[0]: rnn_state[0],
                            self.local_net.state_in[1]: rnn_state[1]
                        })[0][0]
                        v_l, p_l, e_l, g_n, v_n = self._train(rollout, sess, v1)
                        rollout = []

                        sess.run(self.update_local_op)
                    if d:
                        break

                # update network at the end of the episode
                if len(rollout) != 0:
                    v_l, p_l, e_l, g_n, v_n = self._train(rollout, sess, 0.0)

                # episode end
                if running_reward:
                    running_reward = running_reward * 0.99 + ep_reward * 0.01
                else:
                    running_reward = ep_reward

                if ep_count % 10 == 0:
                    print('%s  ep:%d  step:%d  reward:%.3f' %
                          (self.name, ep_count, ep_step_count, running_reward))

                if self.name == 'worker_0':
                    # update global ep
                    _, global_ep = sess.run([
                            self.global_ep.assign_add(1),
                            self.global_ep
                    ])
                    # end condition
                    if global_ep == self.args.max_ep:
                        # this op will stop all threads
                        coord.request_stop()
                    # save model and make gif
                    if global_ep != 0 and global_ep % self.args.save_every == 0:
                        saver.save(
                            sess, self.args.save_path+str(global_ep)+'.cptk')
                ep_count += 1  # update local ep

    def _train(self, rollout, sess, bootstrap_value):
        rollout = np.array(rollout)
        observs, actions, rewards, next_observs, dones, values = rollout.T

        # compute advantages and discounted reward using rewards and value
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, self.gamma)[:-1]

        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + self.gamma * self.value_plus[1:] - \
            self.value_plus[:-1]
        advantages = discount(advantages, self.gamma)

        # update glocal network using gradients from loss
        rnn_state = self.local_net.state_init
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([
            self.local_net.value_loss,
            self.local_net.policy_loss,
            self.local_net.entropy,
            self.local_net.grad_norms,
            self.local_net.var_norms,
            self.local_net.apply_grads
        ], {
            self.local_net.target_v: discounted_rewards,  # for value net
            self.local_net.inputs: np.vstack(observs),
            self.local_net.actions: actions,
            self.local_net.advantages: advantages,  # for policy net
            self.local_net.state_in[0]: rnn_state[0],
            self.local_net.state_in[1]: rnn_state[1]
        })
        return v_l/len(rollout), p_l/len(rollout), e_l/len(rollout), g_n, v_n

    def _update_local_params(self):
        global_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        local_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        update_op = []
        for global_var, local_var in zip(global_vars, local_vars):
            update_op.append(local_var.assign(global_var))

        return update_op
