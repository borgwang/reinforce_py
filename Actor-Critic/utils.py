import numpy as np

def preprocess(obser):
    obser = obser[35:195] # 160x160x3
    obser = obser[::2, ::2, 0] # downsample (80x80)
    obser[obser == 144] = 0
    obser[obser == 109] = 0
    obser[obser != 0] = 1

    return obser.astype(np.float).ravel()
