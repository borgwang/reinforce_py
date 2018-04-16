import scipy.signal
import numpy as np
import random
import tensorflow as tf


def set_global_seeds(env, seed):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env_seeds = np.random.randint(low=0, high=1e6, size=env.num_envs)
    env.set_random_seed(env_seeds)


class RunningMeanStd(object):

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def discount(x, gamma):
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1], axis=0)[::-1]


# ================================================================
# Network components
# ================================================================
def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError

        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def fc(x, out_dim, activation_fn=tf.nn.relu, init_scale=1.0, scope=''):
    with tf.variable_scope(scope):
        in_dim = x.get_shape()[1].value
        w = tf.get_variable('w', [in_dim, out_dim], initializer=ortho_init(init_scale))
        b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        h = activation_fn(z) if activation_fn else z
        return h

# ================================================================
# Tensorflow math utils
# ================================================================
clip = tf.clip_by_value

def sum(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_sum(x, axis=axis, keep_dims=keepdims)

def mean(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_mean(x, axis=axis, keep_dims=keepdims)

def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=keepdims)
    return mean(tf.square(x - meanx), axis=axis, keepdims=keepdims)

def std(x, axis=None, keepdims=False):
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))

def max(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_max(x, axis=axis, keep_dims=keepdims)

def min(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_min(x, axis=axis, keep_dims=keepdims)

def concatenate(arrs, axis=0):
    return tf.concat(axis=axis, values=arrs)

def argmax(x, axis=None):
    return tf.argmax(x, axis=axis)

def switch(condition, then_expression, else_expression):
    """Switches between two operations depending on a scalar value (int or bool).
    Note that both `then_expression` and `else_expression`
    should be symbolic tensors of the *same shape*.

    # Arguments
        condition: scalar tensor.
        then_expression: TensorFlow operation.
        else_expression: TensorFlow operation.
    """
    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: then_expression,
                lambda: else_expression)
    x.set_shape(x_shape)
    return x

# ================================================================
# Math utils
# ================================================================
def explained_variance(pred_y, y):
    """
    Computes fraction of variance that pred_y explains about y.
    Returns 1 - Var[y-pred_y] / Var[y]

    Interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and pred_y.ndim == 1
    var_y = np.var(y)
    return np.nan if var_y == 0 else 1 - np.var(y - pred_y) / var_y
