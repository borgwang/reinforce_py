import os

import gym
import numpy as np

from skimage.color import rgb2gray
from skimage.transform import resize


class Atari(object):
    s_dim = [84, 84, 1]
    a_dim = 3

    def __init__(self, args, record_video=False):
        self.env = gym.make('BreakoutNoFrameskip-v4')
        self.ale = self.env.env.ale  # ale interface
        if record_video:
            video_dir = os.path.join(args.save_path, 'videos')
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            self.env = gym.wrappers.Monitor(
                self.env, video_dir, video_callable=lambda x: True, resume=True)
            self.ale = self.env.env.env.ale

        self.screen_size = Atari.s_dim[:2]  # 84x84
        self.noop_max = 30
        self.frame_skip = 4
        self.frame_feq = 4
        self.s_dim = Atari.s_dim
        self.a_dim = Atari.a_dim

        self.action_space = [1, 2, 3]  # Breakout specify
        self.done = True

    def new_round(self):
        if not self.done:  # dead but not done
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            obs = self.preprocess(obs)
        else:  # terminal
            self.env.reset()
            # No-op
            for _ in range(np.random.randint(1, self.noop_max + 1)):
                obs, _, done, _ = self.env.step(0)
            obs = self.preprocess(obs)
        return obs

    def preprocess(self, observ):
        return resize(rgb2gray(observ), self.screen_size)

    def step(self, action):
        observ, reward, dead = None, 0, False
        for _ in range(self.frame_skip):
            lives_before = self.ale.lives()
            o, r, self.done, _ = self.env.step(self.action_space[action])
            lives_after = self.ale.lives()
            reward += r
            if lives_before > lives_after:
                dead = True
                break
        observ = self.preprocess(o)
        observ = np.reshape(observ, newshape=self.screen_size + [1])
        self.state = np.append(self.state[:, :, 1:], observ, axis=2)

        return self.state, reward, dead, self.done
