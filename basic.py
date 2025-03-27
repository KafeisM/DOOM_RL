#Import vizdoom for game env
import vizdoom as vzd
#Import environment base class from OpenAI Gym
import gymnasium as gym
#Import gym spaces
from gymnasium.spaces import Discrete, Box
#Import opencv
import cv2
#Import vizdoom for game env
from vizdoom import *
#Import random for action sampling
import random
#Import time for sleeping b/w frames
import time
#Import numpy for identity matrix
import numpy as np

from stable_baselines3.common import env_checker
from matplotlib import pyplot as plt

#Import os for file nav
import os
#Import callback class from sb3
from stable_baselines3.common.callbacks import BaseCallback
#import PPO for training
from stable_baselines3 import PPO



#Create a VizDoom environment
class VizDoomEnv(gym.Env):
    #Constructor to initialize the environment
    def __init__(self,render=False):
        super().__init__()

        #Set up game
        self.game = vzd.DoomGame()
        self.game.load_config("./ViZDoom/scenarios/basic.cfg")

        #Render the game
        if render:
            self.game.set_window_visible(True)
        else:
            self.game.set_window_visible(False)

        #Set the Game scenario
        self.game.init()

        #Create the acion space and observation space
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8) #Grayscale image, 100x160,
        self.action_space = Discrete(3) #3 actions: turn left, turn right, shoot

    #Define how the environment steps
    def step(self, action):
        actions = np.identity(3, dtype=int)
        reward = self.game.make_action(actions[action], 4)

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info":info}
        done = self.game.is_episode_finished()
        truncated = False

        return state, reward, done, truncated, info

    #Define how to render the game or environment
    def render(self):
        pass

    #What happen when we start a new game
    def reset(self, seed=None):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        ammo = self.game.get_state().game_variables[0]
        info = {"ammo":ammo}
        return self.grayscale(state), info

    #Grayscale the game frame and resize it
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state

    #Call to close down the game
    def close(self):
        self.game.close()

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


def main():
    CHECKPOINT_DIR = 'train/train_basic'
    LOG_DIR = 'logs/log_basic'

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

    env = VizDoomEnv()
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048)
    model.learn(total_timesteps=100000, callback=callback)


if __name__ == "__main__":
    main()