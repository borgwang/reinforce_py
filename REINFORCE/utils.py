import numpy as np

# preprocess function
def preprocess(obser):
    '''preprocess 210x160x3 frame into 6400(80x80) flat vector'''
    obser = obser[35:195] # 160x160x3
    obser = obser[::2, ::2, 0] # downsample (80x80)
    obser[obser == 144] = 0
    obser[obser == 109] = 0
    obser[obser != 0] = 1

    return obser.astype(np.float).ravel()
