from vizdoom import *
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
import os.path

class BaseVizDoomEnv(gym.Env):
    """Clase base para entornos VizDoom simples"""
    
    def __init__(self, scenario_path, num_actions, render=False):
        super().__init__()

        print("Loading scenario:", scenario_path)

        # Set up game
        self.game = DoomGame()  # type: ignore
        self.game.load_config(scenario_path)

        # Render the game
        if render:
            self.game.set_window_visible(True)
        else:
            self.game.set_window_visible(False)

        self.game.set_screen_format(ScreenFormat.GRAY8)  # type: ignore
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)  # type: ignore
        
        # Inicializamos el juego
        self.game.init()
        
        # Guardamos la cantidad de acciones
        self.num_actions = num_actions if num_actions > 0 else self.game.get_available_buttons_size()

        # Create the action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(120, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(self.num_actions)

        print(f"Variables de juego disponibles: {self.game.get_available_game_variables()}")
        print(f"Acciones disponibles: {self.game.get_available_buttons()}")
        print(f"Formato de pantalla: {self.game.get_screen_format()}")

    def step(self, action):
        actions = np.identity(self.num_actions, dtype=int)
        reward = self.game.make_action(actions[action], 4)

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = np.expand_dims(state, axis=-1)  # Añadimos dimensión de canal
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info": info}
        terminated = self.game.is_episode_finished()
        truncated = False

        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        state = np.expand_dims(state, axis=-1)  # Añadimos dimensión de canal
        ammo = self.game.get_state().game_variables[0]
        info = {"ammo": ammo}
        return state, info

    def render(self):
        # Ya está implementado con la ventana visible
        pass

    def close(self):
        self.game.close()