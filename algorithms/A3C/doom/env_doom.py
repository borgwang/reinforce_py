import numpy as np

from vizdoom import *


class Doom(object):
    '''Wrapper for Doom environment. Gym-style interface'''
    def __init__(self, visiable=False):
        self.env = self._setup(visiable)
        self.state_dim = 84 * 84 * 1
        self.action_dim = 3
        # Identity bool matrix, transfer action to bool one-hot
        self.bool_onehot = np.identity(self.action_dim, dtype=bool).tolist()

    def _setup(self, visiable):
        # setting up Doom environment
        env = DoomGame()
        env.set_doom_scenario_path('basic.wad')
        env.set_doom_map('map01')
        env.set_screen_resolution(ScreenResolution.RES_160X120)
        env.set_screen_format(ScreenFormat.GRAY8)
        env.set_render_hud(False)
        env.set_render_crosshair(False)
        env.set_render_weapon(True)
        env.set_render_decals(False)
        env.set_render_particles(False)
        env.add_available_button(Button.MOVE_LEFT)
        env.add_available_button(Button.MOVE_RIGHT)
        env.add_available_button(Button.ATTACK)
        env.add_available_game_variable(GameVariable.AMMO2)
        env.add_available_game_variable(GameVariable.POSITION_X)
        env.add_available_game_variable(GameVariable.POSITION_Y)
        env.set_episode_timeout(300)
        env.set_episode_start_time(10)
        env.set_sound_enabled(False)
        env.set_living_reward(-1)
        env.set_mode(Mode.PLAYER)
        env.set_window_visible(visiable)
        env.init()

        return env

    def _get_state(self):
        return self.env.get_state().screen_buffer

    def reset(self):
        self.env.new_episode()
        return self._get_state()

    def step(self, action):
        action = self.bool_onehot[action]  # e.g. [False, True, False]

        curr_observ = self._get_state()
        reward = self.env.make_action(action)
        done = self.env.is_episode_finished()
        if done:
            next_observ = curr_observ
        else:
            next_observ = self._get_state()

        return next_observ, reward, done
