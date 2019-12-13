import numpy as np
import scipy.signal
import tensorflow as tf


def preprocess(frame):
    s = frame[10: -10, 30: -30]
    s = scipy.misc.imresize(s, [84, 84])
    s = np.reshape(s, [np.prod(s.shape)]) / 255.0
    return s


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def normalized_columns_initializer(std=1.0):

    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def print_net_params_number():
    total_parameters = 0
    for v in tf.trainable_variables():
        shape = v.get_shape()
        param_num = 1
        for d in shape:
            param_num *= d.value
        print(v.name, ' ', shape, ' param nums: ', param_num)
        total_parameters += param_num
    print('\nTotal nums of parameters: %d\n' % total_parameters)
