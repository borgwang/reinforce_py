import numpy as np


class GridWorld:

    def __init__(self):
        self.env_w = 10
        self.env_h = 10
        self.num_s = self.env_w * self.env_h
        self.num_a = 4

        r = np.zeros(self.num_s)
        w = np.zeros(self.num_s)

        self.target = np.array([27])
        self.bomb = np.array([16, 25, 26, 28, 36, 40, 41, 48, 49, 64])
        # make some walls
        self.wall = np.array([22, 32, 42, 52, 43, 45, 46, 47, 37])
        r[self.target] = 10
        r[self.bomb] = -1
        r[self.wall] = 0
        w[self.wall] = 1

        self.W = w
        self.R = r  # reward
        self.terminal = np.array(self.target)

    def rewards(self, s, a):
        return self.R[s]

    def allow_actions(self, s):
        # return allow actions in state s
        x = self.get_pos(s)[0]
        y = self.get_pos(s)[1]
        allow_a = np.array([], dtype='int')
        if y > 0 and self.W[s-self.env_w] != 1:
            allow_a = np.append(allow_a, 0)
        if y < self.env_h-1 and self.W[s+self.env_w] != 1:
            allow_a = np.append(allow_a, 1)
        if x > 0 and self.W[s-1] != 1:
            allow_a = np.append(allow_a, 2)
        if x < self.env_w-1 and self.W[s+1] != 1:
            allow_a = np.append(allow_a, 3)
        return allow_a

    def get_pos(self, s):
        # transform to coordinate (x, y)
        x = s % self.env_h
        y = s / self.env_w
        return x, y

    def next_state(self, s, a):
        # return next state in state s taking action a
        # in this deterministic environment it returns a certain state ns
        ns = 0
        if a == 0:
            ns = s - self.env_w
        if a == 1:
            ns = s + self.env_w
        if a == 2:
            ns = s - 1
        if a == 3:
            ns = s + 1
        return ns

    def is_terminal(self, s):
        return True if s in self.terminal else False

    def reset(self):
        return 0  # init state
