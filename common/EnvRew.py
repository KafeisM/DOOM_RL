import gymnasium as gym
import numpy as np
import cv2
from gymnasium.spaces import Box, Discrete
from vizdoom import DoomGame

class VizDoomEnv(gym.Env):
    def __init__(self, config_path, render=False):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config(config_path)
        self.game.set_window_visible(render)
        self.game.init()

        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(self.game.get_available_buttons_size())

    def step(self, action):
        reward = self.game.make_action([int(i == action) for i in range(self.action_space.n)], 4)
        done = self.game.is_episode_finished()
        obs = np.zeros((100, 160, 1), dtype=np.uint8) if done else self._get_obs()
        info = {}
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        return self._get_obs(), {}

    def _get_obs(self):
        img = self.game.get_state().screen_buffer
        gray = cv2.cvtColor(np.moveaxis(img, 0, -1), cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_AREA)
        return resized[..., np.newaxis]

    def close(self):
        self.game.close()
