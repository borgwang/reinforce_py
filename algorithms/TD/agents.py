import numpy as np
import random

from utils import draw_episode_steps
from utils import draw_grid


class TDAgent(object):

    def __init__(self, env, epsilon, gamma, alpha=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon  # explore & exploit
        self.init_epsilon = epsilon

        self.P = np.zeros((self.env.num_s, self.env.num_a))

        self.V = np.zeros(self.env.num_s)
        self.Q = np.zeros((self.env.num_s, self.env.num_a))

        self.step_set = []  # store steps of each episode
        self.avg_step_set = []  # store average steps of each 100 episodes
        self.episode = 1
        self.step = 0
        self.max_episodes = 5000

        # initialize random policy
        for s in range(self.env.num_s):
            poss = self.env.allow_actions(s)
            for a in poss:
                self.P[s][a] = 1.0 / len(poss)

        self.curr_s = None
        self.curr_a = None

    def predict(self, episode=1000):
        for e in range(episode):
            curr_s = self.env.reset()  # new episode
            while not self.env.is_terminal(curr_s):  # for every time step
                a = self.select_action(curr_s, policy='greedy')
                r = self.env.rewards(curr_s, a)
                next_s = self.env.next_state(curr_s, a)
                self.V[curr_s] += self.alpha \
                    * (r+self.gamma*self.V[next_s] - self.V[curr_s])
                curr_s = next_s
        # result display
        draw_grid(self.env, self, p=True, v=True, r=True)

    def control(self, method):
        assert method in ("qlearn", "sarsa")

        if method == "qlearn":
            agent = Qlearn(self.env, self.epsilon, self.gamma)
        else:
            agent = SARSA(self.env, self.epsilon, self.gamma)

        while agent.episode < self.max_episodes:
            agent.learn(agent.act())

        # result display
        draw_grid(self.env, agent, p=True, v=True, r=True)
        # draw episode steps
        draw_episode_steps(agent.avg_step_set)

    def update_policy(self):
        # update according to Q value
        poss = self.env.allow_actions(self.curr_s)
        # Q values of all allowed actions
        qs = self.Q[self.curr_s][poss]
        q_maxs = [q for q in qs if q == max(qs)]
        # update probabilities
        for i, a in enumerate(poss):
            self.P[self.curr_s][a] = \
                1.0 / len(q_maxs) if qs[i] in q_maxs else 0.0

    def select_action(self, state, policy='egreedy'):
        poss = self.env.allow_actions(state)  # possible actions
        if policy == 'egreedy' and random.random() < self.epsilon:
            a = random.choice(poss)
        else:  # greedy action
            pros = self.P[state][poss]  # probabilities for possible actions
            best_a_idx = [i for i, p in enumerate(pros) if p == max(pros)]
            a = poss[random.choice(best_a_idx)]
        return a


class SARSA(TDAgent):

    def __init__(self, env, epsilon, gamma):
        super(SARSA, self).__init__(env, epsilon, gamma)
        self.reset_episode()

    def act(self):
        s = self.env.next_state(self.curr_s, self.curr_a)
        a = self.select_action(s, policy='egreedy')
        r = self.env.rewards(self.curr_s, self.curr_a)
        r -= 0.01  # a bit negative reward for every step
        return [self.curr_s, self.curr_a, r, s, a]

    def learn(self, exp):
        s, a, r, n_s, n_a = exp

        if self.env.is_terminal(s):
            target = r
        else:
            target = r + self.gamma * self.Q[n_s][n_a]
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

        # update policy
        self.update_policy()

        if self.env.is_terminal(s):
            self.V = np.sum(self.Q, axis=1)
            print('episode %d step: %d epsilon: %f' %
                  (self.episode, self.step, self.epsilon))
            self.reset_episode()
            self.epsilon -= self.init_epsilon / 10000
            # record per 100 episode
            if self.episode % 100 == 0:
                self.avg_step_set.append(
                    np.sum(self.step_set[self.episode-100: self.episode])/100)
        else:  # shift state-action pair
            self.curr_s = n_s
            self.curr_a = n_a
            self.step += 1

    def reset_episode(self):
        # start a new episode
        self.curr_s = self.env.reset()
        self.curr_a = self.select_action(self.curr_s, policy='egreedy')
        self.episode += 1
        self.step_set.append(self.step)
        self.step = 0


class Qlearn(TDAgent):

    def __init__(self, env, epsilon, gamma):
        super(Qlearn, self).__init__(env, epsilon, gamma)
        self.reset_episode()

    def act(self):
        a = self.select_action(self.curr_s, policy='egreedy')
        s = self.env.next_state(self.curr_s, a)
        r = self.env.rewards(self.curr_s, a)
        r -= 0.01
        return [self.curr_s, a, r, s]

    def learn(self, exp):
        s, a, r, n_s = exp

        # Q-learning magic
        if self.env.is_terminal(s):
            target = r
        else:
            target = r + self.gamma * max(self.Q[n_s])
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

        self.update_policy()
        # shift to next state
        if self.env.is_terminal(s):
            self.V = np.sum(self.Q, axis=1)
            print('episode %d step: %d' % (self.episode, self.step))
            self.reset_episode()
            self.epsilon -= self.init_epsilon / self.max_episodes
            # record per 100 episode
            if self.episode % 100 == 0:
                self.avg_step_set.append(
                    np.sum(self.step_set[self.episode-100: self.episode])/100)
        else:
            self.curr_s = n_s
            self.step += 1

    def reset_episode(self):
        self.curr_s = self.env.reset()
        self.episode += 1
        self.step_set.append(self.step)
        self.step = 0
