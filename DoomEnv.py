import vizdoom as vzd
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import cv2
import numpy as np

class BaseVizDoomEnv(gym.Env):
    """Clase base para entornos VizDoom"""
    
    def __init__(self, scenario_path,num_actions ,render=False):
        super().__init__()

        print("Loading scenario:", scenario_path)

        # Set up game
        self.game = vzd.DoomGame()
        self.game.load_config(scenario_path)

        self.num_actions = num_actions

        # Render the game
        if render:
            self.game.set_window_visible(True)
        else:
            self.game.set_window_visible(False)

        # Set the Game scenario
        self.game.init()

        # Create the action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8)
        self.action_space = Discrete(num_actions)  # 3 actions: turn left, turn right, shoot

    def step(self, action):
        actions = np.identity(self.num_actions, dtype=int)
        reward = self.game.make_action(actions[action], 4)

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info": info}
        done = self.game.is_episode_finished()
        truncated = False

        return state, reward, done, truncated, info

    def render(self):
        pass

    def reset(self, seed=None):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        ammo = self.game.get_state().game_variables[0]
        info = {"ammo": ammo}
        return self.grayscale(state), info

    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state

    def close(self):
        self.game.close()