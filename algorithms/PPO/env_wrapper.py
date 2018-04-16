import time
import csv
import json
import gym
from gym.core import Wrapper
import os.path as osp
import numpy as np

from utils import RunningMeanStd


class BaseVecEnv(object):
    """
    Vectorized environment base class
    """

    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, dones)
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset all environments
        """
        raise NotImplementedError

    def close(self):
        pass

    def set_random_seed(self, seed):
        raise NotImplementedError


class VecEnv(BaseVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')

    def step(self, action_n):
        results = [env.step(a) for (a, env) in zip(action_n, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def render(self):
        self.envs[0].render()

    @property
    def num_envs(self):
        return len(self.envs)


class VecEnvNorm(BaseVecEnv):

    def __init__(self, venv, ob=True, ret=True,
                 clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        self.venv = venv
        self._ob_space = venv.observation_space
        self._ac_space = venv.action_space
        self.ob_rms = RunningMeanStd(shape=self._ob_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, vac):
        obs, rews, news, infos = self.venv.step(vac)
        self.ret = self.ret * self.gamma + rews
        # normalize observations
        obs = self._norm_ob(obs)
        # normalize rewards
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon),
                           -self.cliprew, self.cliprew)
        return obs, rews, news, infos

    def _norm_ob(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        obs = self.venv.reset()
        return self._norm_ob(obs)

    def set_random_seed(self, seeds):
        for env, seed in zip(self.venv.envs, seeds):
            env.seed(int(seed))

    @property
    def action_space(self):
        return self._ac_space

    @property
    def observation_space(self):
        return self._ob_space

    def close(self):
        self.venv.close()

    def render(self):
        self.venv.render()

    @property
    def num_envs(self):
        return self.venv.num_envs


class Monitor(Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=()):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if osp.isdir(filename):
                    filename = osp.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            self.f = open(filename, "wt")
            self.f.write('#%s\n'%json.dumps({"t_start": self.tstart, "gym_version": gym.__version__,
                "env_id": env.spec.id if env.spec else 'Unknown'}))
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+reset_keywords)
            self.logger.writeheader()

        self.reset_keywords = reset_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_steps = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()

    def _reset(self, **kwargs):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError('Expected you to pass kwarg %s into reset'%k)
            self.current_reset_info[k] = v
        return self.env.reset(**kwargs)

    def _step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            epinfo.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(epinfo)
                self.f.flush()
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            info['episode'] = epinfo
        self.total_steps += 1
        return (ob, rew, done, info)

    def close(self):
        if self.f is not None:
            self.f.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths


def make_env():
    def env_fn():
        env = gym.make(args.env)
        env = Monitor(env, logger.get_dir())
        return env
    env = VecEnv([env_fn] * args.n_envs)
    env = VecEnvNorm(env)
    return env
