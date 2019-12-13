import numpy as np
import gym
from gym.spaces import Discrete, Box


# policies
class DiscreteAction(object):

    def __init__(self, theta, ob_space, ac_space):
        ob_dim = ob_space.shape[0]
        ac_dim = ac_space.n
        self.W = theta[0: ob_dim * ac_dim].reshape(ob_dim, ac_dim)
        self.b = theta[ob_dim * ac_dim:].reshape(1, ac_dim)

    def act(self, ob):
        y = np.dot(ob, self.W) + self.b
        a = np.argmax(y)
        return a


class ContinuousAction(object):

    def __init__(self, theta, ob_space, ac_space):
        self.ac_space = ac_space
        ob_dim = ob_space.shape[0]
        ac_dim = ac_space.shape[0]
        self.W = theta[0: ob_dim * ac_dim].reshape(ob_dim, ac_dim)
        self.b = theta[ob_dim * ac_dim:]

    def act(self, ob):
        y = np.dot(ob, self.W) + self.b
        a = np.clip(y, self.ac_space.low, self.ac_space.high)
        return a


def run_episode(policy, env, render=False):
    max_steps = 1000
    total_rew = 0
    ob = env.reset()
    for t in range(max_steps):
        a = policy.act(ob)
        ob, reward, done, _info = env.step(a)
        total_rew += reward
        if render and t % 3 == 0:
            env.render()
        if done:
            break
    return total_rew


def make_policy(params):
    if isinstance(env.action_space, Discrete):
        return DiscreteAction(params, env.observation_space, env.action_space)
    elif isinstance(env.action_space, Box):
        return ContinuousAction(
            params, env.observation_space, env.action_space)
    else:
        raise NotImplementedError


def eval_policy(params):
    policy = make_policy(params)
    reward = run_episode(policy, env)
    return reward


env = gym.make('CartPole-v0')
num_iter = 100
batch_size = 25
elite_frac = 0.2
num_elite = int(batch_size * elite_frac)

if isinstance(env.action_space, Discrete):
    dim_params = (env.observation_space.shape[0] + 1) * env.action_space.n
elif isinstance(env.action_space, Box):
    dim_params = (env.observation_space.shape[0] + 1) \
        * env.action_space.shape[0]
else:
    raise NotImplementedError

params_mean = np.zeros(dim_params)
params_std = np.ones(dim_params)

for i in range(num_iter):
    # sample parameter vectors (multi-variable gaussian distribution)
    sample_params = np.random.multivariate_normal(
                        params_mean, np.diag(params_std), size=batch_size)
    # evaluate sample policies
    rewards = [eval_policy(params) for params in sample_params]

    # 'elite' policies
    elite_idxs = np.argsort(rewards)[batch_size - num_elite: batch_size]
    elite_params = [sample_params[i] for i in elite_idxs]

    # move current policy towards elite policies
    params_mean = np.mean(np.asarray(elite_params), axis=0)
    params_std = np.std(np.asarray(elite_params), axis=0)

    # logging
    print('Ep %d: mean score: %8.3f. max score: %4.3f' %
          (i, np.mean(rewards), np.max(rewards)))
    print('Eval reward: %.4f' %
          run_episode(make_policy(params_mean), env, render=True))
