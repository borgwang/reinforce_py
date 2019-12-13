import time

import numpy as np
import scipy.signal
import tensorflow as tf


def reward_discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def ortho_init(scale=1.0):

    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


def print_params_nums():
    total_parameters = 0
    for v in tf.trainable_variables():
        shape = v.get_shape()
        param_num = 1
        for d in shape:
            param_num *= d.value
        print(v.name, ' ', shape, ' param nums: ', param_num)
        total_parameters += param_num
    print('\nTotal nums of parameters: %d\n' % total_parameters)


def print_time_cost(start_time):
    t_c = time.gmtime(time.time() - start_time)
    print('Time cost ------ %dh %dm %ds' %
          (t_c.tm_hour, t_c.tm_min, t_c.tm_sec))
