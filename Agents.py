import numpy as np
import random
from utils import draw_grid, draw_episode_steps

class DPAgent(object):
    def __init__(self,env, gamma=1):
        self.env = env
        self.gamma = gamma
        self.ns = self.env.num_s
        self.na = self.env.num_a
        self.P = np.zeros((self.ns, self.na)) # store probability in a 1-D array
        self.V = np.zeros(self.ns)
        self.converge = False
        # initialize random policy
        for s in range(self.ns):
            poss = self.env.allow_actions(s)
            for a in poss:
                self.P[s][a] = 1.0 / len(poss)

    def evaluate_policy(self):
        # one sweep
        new_V = np.zeros(self.ns)
        for s in range(self.ns):
            if self.env.W[s] == 1: continue
            new_v = 0
            poss = self.env.allow_actions(s) # possible actions
            for a in poss:
                pro = self.P[s][a]  # probability of taking action a
                next_s = self.env.next_state(s,a)
                new_v += pro * (self.env.rewards(s, a) + self.gamma * self.V[next_s])
            new_V[s] = new_v
        if np.sum(np.square(np.abs(self.V)-np.abs(new_V))) < 0.0001:
            self.converge = True
            draw_grid(self.env, self, p=True, v=True, r=True)
        self.V = new_V

    def improve_policy(self):
        # greedy
        for s in range(self.ns):
            if self.env.W[s] == 1: continue
            v_l = []
            poss = self.env.allow_actions(s)
            for a in poss:
                pro = self.P[s][a]
                next_s = self.env.next_state(s,a)
                v_a = pro * (self.env.rewards(s, a) + self.gamma * self.V[next_s])
                v_l.append(v_a)
            v_maxs = [v for v in v_l if v == max(v_l)]
            v_max = v_maxs[0]
            # update probabilities
            for i,a in enumerate(poss):
                self.P[s][a] = 1.0 / len(v_maxs) if v_l[i] in v_maxs else 0.0


class MCAgent(object):
    def __init__(self, env, epsilon, gamma=0.9, alpha=0.1):
        self.env = env
        # hyperparameters
        self.epsilon = epsilon
        self.init_epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        self.P = np.zeros((env.num_s, env.num_a))  # store probability
        self.V = np.zeros(env.num_s)
        self.Q = np.zeros((env.num_s, env.num_a))
        self.converge = False
        self.max_episodes = 10000
        self.step = 0

        # initialize random policy
        for s in range(env.num_s):
            poss = self.env.allow_actions(s)
            for a in poss:
                self.P[s][a] = 1.0 / len(poss)

    def predict_v(self):
        # Monte-Carlo estimation of state value
        for e in range(self.max_episodes):
            curr_s = self.env.reset()
            ss,rs = [],[]
            # act and save the trajectory
            while not self.env.is_terminal(curr_s):
                a = self.select_action(curr_s, self.P)
                r, next_s = self.act(curr_s, a)
                ss.append(curr_s)
                rs.append(r)
                curr_s = next_s
            gs = []
            # calculate return for each state
            for i,r in enumerate(rs[::-1]):
                if i == 0:
                    # terminal state
                    gs.insert(0, r)
                else:
                    gs.insert(0, r+self.gamma*gs[0])
            # remove duplicate elements (first visit)
            f_s,f_G = [],[]
            for s,g in zip(ss,gs):
                if s not in f_s:
                    f_s.append(s)
                    f_G.append(g)
            # update v for all states in the episode
            self.V[f_s] += self.alpha * (f_G - self.V[f_s])
        draw_grid(self.env, self, p=True, v=True, r=True)

    def control(self):
        for e in range(self.max_episodes):
            # new episode
            self.curr_s = self.env.reset()
            self.step = 0
            sas, rs = [], []

            # act and save the trajectory
            while not self.env.is_terminal(self.curr_s) and self.step < 200:
                a = self.select_action(self.curr_s, policy='egreedy')
                r, next_s = self.act(self.curr_s, a)
                sas.append([self.curr_s,a])
                rs.append(r)
                self.curr_s = next_s
                self.step += 1

            a = self.select_action(self.curr_s, policy='egreedy')
            r, next_s = self.act(self.curr_s, a)
            sas.append([self.curr_s,a])
            rs.append(r)
            gs = np.zeros_like(rs, dtype=np.float)
            running_add = 0
            for t in reversed(range(len(rs))):
                if rs[t] != 0:
                    running_add = rs[t]
                    for i in reversed(range(t+1)):
                        running_add = running_add * self.gamma
                        gs[i] += running_add
            # calculate return for each (s,a)
            # gs = []
            # for i,r in enumerate(rs[::-1]):
            #     r = r if i ==0 else r+self.gamma*gs[0]
            #     gs.insert(0, r)

            # remove duplicate elements (first visit)
            f_sa,f_G = [],[]
            for sa,g in zip(sas,gs):
                if sa not in f_sa:
                    f_sa.append(sa)
                    f_G.append(g)

            for sa, g in zip(f_sa, f_G):
                # update Q network
                self.Q[sa[0]][sa[1]] += self.alpha * (g - self.Q[sa[0]][sa[1]])
                # update policy
                self.update_policy(sa[0])

            self.epsilon -= self.init_epsilon / self.max_episodes
            print "episode %d step: %d epsilon: %.4f" % (e, self.step, self.epsilon)

        # resutl display
        self.V = np.sum(self.Q, axis=1)
        draw_grid(self.env, self, p=True, v=True)

    def select_action(self, state, policy='egreedy'):
        poss = self.env.allow_actions(state) # possiable actions
        if policy == 'egreedy' and random.random() < self.epsilon:  # random action
            a = random.choice(poss)
        else:  # greedy action
            pros = self.P[state][poss] # probobilities for possiable actions
            best_a_idx = [i for i,p in enumerate(pros) if p == max(pros)] # action(s) that have the max prob
            a = poss[random.choice(best_a_idx)]
        return a

    def act(self, s, a):
        # Exxecute action a in state s. Return reward r and next state next_s
        r = self.env.rewards(s, a)
        next_s = self.env.next_state(s, a)
        return r, next_s

    def update_policy(self,state):
        # update according to Q value
        poss = self.env.allow_actions(state)
        # Q values of all allowed actions
        qs = self.Q[state][poss] # used to be a hidden bug here...
        q_maxs = [q for q in qs if q == max(qs)]
        # update probs
        for i,a in enumerate(poss):
            self.P[state][a] = 1.0 / len(q_maxs) if qs[i] in q_maxs else 0.0


class TDAgent(object):
    def __init__(self, env, epsilon, gamma, alpha=0.05, lamda=0):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon  # explore & exploit
        self.init_episilon = epsilon
        self.lamda = lamda

        self.V = np.zeros(self.env.num_s)
        self.P = np.zeros((self.env.num_s, self.env.num_a))
        self.Q = np.zeros((self.env.num_s, self.env.num_a))
        self.Z = np.zeros((self.env.num_s, self.env.num_a))

        self.step_set = [] # store steps of each episode
        self.avg_step_set = [] # store average steps of each 100 episodes
        self.episode = 1
        self.step = 0
        self.max_episodes = 10000

        # initialize random policy
        for s in range(self.env.num_s):
            poss = self.env.allow_actions(s)
            for a in poss:
                self.P[s][a] = 1.0 / len(poss)

    def predict(self, episode=1000):
        for e in xrange(episode):
            curr_s = self.env.reset()  # new episode
            while not self.env.is_terminal(curr_s): # for every time step
                a = self.select_action(curr_s, policy='greedy')
                r = self.env.rewards(curr_s, a)
                next_s = self.env.next_state(curr_s, a)
                self.V[curr_s] += self.alpha*(r+self.gamma*self.V[next_s] - self.V[curr_s])
                curr_s = next_s
        # result display
        draw_grid(self.env, self, p=True, v=True, r=True)

    def control(self, method):
        assert method == 'qlearn' or method == 'sarsa'

        if method == 'qlearn':
            agent = Qlearn(self.env, self.epsilon, self.gamma)
        elif method == 'sarsa':
            agent = SARSA(self.env, self.epsilon, self.gamma)

        while agent.episode < self.max_episodes:
            agent.learn(agent.act())

        # resutl display
        draw_grid(self.env, agent, p=True, v=True, r=True)
        # draw episode steps
        draw_episode_steps(agent.avg_step_set)


    def update_policy(self):
        # update according to Q value
        poss = self.env.allow_actions(self.curr_s)
        # Q values of all allowed actions
        qs = self.Q[self.curr_s][poss]
        q_maxs = [q for q in qs if q == max(qs)]
        # update probs
        for i,a in enumerate(poss):
            self.P[self.curr_s][a] = 1.0 / len(q_maxs) if qs[i] in q_maxs else 0.0

    def select_action(self, state, policy='egreedy'):
        poss = self.env.allow_actions(state) # possiable actions
        if policy == 'egreedy' and random.random() < self.epsilon:  # random action
            a = random.choice(poss)
        else:  # greedy action
            pros = self.P[state][poss] # probobilities for possiable actions
            best_a_idx = [i for i,p in enumerate(pros) if p == max(pros)] # action(s) that have the max prob
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
        r -= 0.01 # a bit negative reward for every step
        return [self.curr_s, self.curr_a, r, s, a]

    def learn(self, exp):
        s, a, r, n_s, n_a = exp
        target = r if self.env.is_terminal(s) else r+self.gamma*self.Q[n_s][n_a]
        error = target - self.Q[s][a]

        self.Z[s][a] += 1.0
        for _s in xrange(self.env.num_s):
            for _a in xrange(self.env.num_a):
                self.Q[_s][_a] += self.alpha * error * self.Z[_s][_a]
                self.Z[_s][_a] *= self.lamda * self.gamma

        # update policy
        self.update_policy()

        if self.env.is_terminal(s):
            self.V = np.sum(self.Q, axis=1)
            print "episode %d step: %d epsilon: %f" % (self.episode, self.step, self.epsilon)
            self.reset_episode()
            self.epsilon -= self.init_episilon / 10000
        else: # shift state-action pair
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
        target = r if self.env.is_terminal(s) else r+self.gamma*max(self.Q[n_s])
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

        self.update_policy()
        # shift to next state
        if self.env.is_terminal(s):
            self.V = np.sum(self.Q, axis=1)
            print "episode %d step: %d" % (self.episode, self.step)
            self.reset_episode()
            self.epsilon -= self.init_episilon / self.max_episodes
            # record per 100 episode
            if self.episode % 100 == 0:
                self.avg_step_set.append(np.sum(self.step_set[self.episode-100:self.episode])/100)
        else:
            self.curr_s = n_s
            self.step += 1

    def reset_episode(self):
        self.curr_s = self.env.reset()
        self.episode += 1
        self.step_set.append(self.step)
        self.step = 0
