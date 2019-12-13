import numpy as np


def preprocess(obs):
    obs = obs[35:195]  # 160x160x3
    obs = obs[::2, ::2, 0]  # down sample (80x80)
    obs[obs == 144] = 0
    obs[obs == 109] = 0
    obs[obs != 0] = 1
    return obs.astype(np.float).ravel()
