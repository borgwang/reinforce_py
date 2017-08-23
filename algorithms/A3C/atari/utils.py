import tensorflow as tf
import scipy.signal
import numpy as np
import time


def reward_discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def print_params_nums():
    total_parameters = 0
    for v in tf.trainable_variables():
        shape = v.get_shape()
        param_num = 1
        for d in shape:
            param_num *= d.value
        print v.name, ' ', shape, ' param nums: ', param_num
        total_parameters += param_num
    print '\nTotal nums of parameters: %d\n' % total_parameters


def print_time_cost(start_time):
    t_c = time.gmtime(time.time() - start_time)
    print 'Time cost ------ %dh %dm %ds' % (t_c.tm_hour, t_c.tm_min, t_c.tm_sec)
