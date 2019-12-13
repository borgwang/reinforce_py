import matplotlib.pyplot as plt
import numpy as np


def draw_grid(env, agent, p=True, v=False, r=False):
    '''
    Draw the policy(|value|reward setting) at the command prompt.
    '''
    arrows = [u'\u2191', u'\u2193', u'\u2190', u'\u2192']
    cliff = u'\u25C6'
    sign = {0: '-', 10: u'\u2713', -1: u'\u2717'}

    tp = []  # transform policy
    for s in range(env.num_s):
        for a in range(env.num_a):
            tp.append(agent.P[s][a])
    best = []  # best action for each state at the moment
    for i in range(0, len(tp), env.num_a):
        a = tp[i:i+env.num_a]
        ba = np.argsort(-np.array(a))[0]
        best.append(ba)
    if r:
        print('\n')
        print('Environment setting:', end=' ')
        for i, r in enumerate(env.R):
            if i % env.env_w == 0:
                print('\n')
            if env.W[i] > 0:
                print('%1s' % cliff, end=' ')
            else:
                print('%1s' % sign[r], end=' ')
        print('\n')
    if p:
        print('Trained policy:', end=' ')
        for i, a in enumerate(best):
            if i % env.env_w == 0:
                print('\n')
            if env.W[i] == 1:
                print('%s' % cliff, end=' ')
            elif env.R[i] == 1:
                print('%s' % u'\u272A', end=' ')
            else:
                print('%s' % arrows[a], end=' ')
        print('\n')
    if v:
        print('Value function for each state:', end=' ')
        for i, v in enumerate(agent.V):
            if i % env.env_w == 0:
                print('\n')
            if env.W[i] == 1:
                print(' %-2s ' % cliff, end=' ')
            elif env.R[i] == 1:
                print('[%.1f]' % v, end=' ')
            else:
                print('%4.1f' % v, end=' ')
        print('\n')


def draw_episode_steps(avg_step_set):
    plt.plot(np.arange(len(avg_step_set)), avg_step_set)
    plt.title('steps per episode')
    plt.xlabel('episode')
    plt.ylabel('steps')
    plt.axis([0, 80, 0, 200])
    plt.show()
