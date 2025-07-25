from vizdoom import *
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
import os.path


class VizDoomReward(Env):
    def __init__(self, scenario_path, render=False, seed=None):
        super().__init__()
        self.game = DoomGame() #type: ignore

        # Aplica semilla antes de cargar la configuración y de init
        if seed is not None:
            print(f"[DEBUG] Applying seed before load_config/init: {seed}")
            self.game.set_seed(seed)
            print(f"[SEED] DoomGame seed set to {seed} (pre-load)")

        self.game.load_config(scenario_path)

        # Render the game
        if render:
            self.game.set_window_visible(True)
        else:
            self.game.set_window_visible(False)
        
        #self.game.set_screen_format(ScreenFormat.GRAY8) #type: ignore
        #self.game.set_screen_resolution(ScreenResolution.RES_160X120) #type: ignore
        #self.game.set_available_game_variables([GameVariable.SELECTED_WEAPON_AMMO, GameVariable.HEALTH, GameVariable.KILLCOUNT])  # type: ignore
        #self.game.set_doom_skill(5)

        self.game.init()

        self.ammo = self.game.get_state().game_variables[0]
        self.killcount = 0
        self.health = 100
        
        self.observation_space = Box(low=0, high=255, shape=(120, 160, 1), dtype=np.uint8)  # Cambiado a formato HWC
        self.action_space = Discrete(self.game.get_available_buttons_size())

        print(f"Variables de juego disponibles: {self.game.get_available_game_variables()}")
        print(f"Acciones disponibles: {self.game.get_available_buttons()}")
        print(f"Formato de pantalla: {self.game.get_screen_format()}")

    def step(self, action):
        buttons = self.game.get_available_buttons_size()
        actions = np.identity(buttons)
        movement_reward = (self.game.make_action(actions[action], 4)) / 5
        reward = 0 
        
        terminated = False
        truncated = False
        
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            state = np.expand_dims(state, axis=-1)  # Añadimos dimensión de canal
            game_variables = self.game.get_state().game_variables
            ammo, health, killcount = game_variables
            # Calculate reward deltas
            killcount_delta = killcount - self.killcount
            self.killcount = killcount

            health_delta = health - self.health
            self.health = health

            ammo_delta = ammo - self.ammo
            self.ammo = ammo

            if health_delta < 0:
                health_reward = -5
            else:
                health_reward = 0

            if ammo_delta == 0:
                ammo_reward = 0
            else:
                ammo_reward = ammo_delta * 0.5
        
            if killcount_delta > 0:
                killcount_reward = killcount_delta * 100
            else:
                killcount_reward = 0

            reward = movement_reward + health_reward + ammo_reward + killcount_reward
            reward = reward / 1000
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)
            
        terminated = self.game.is_episode_finished()
        # En VizDoom no hay realmente un concepto de truncado, pero lo dejamos por compatibilidad
        truncated = False
        
        info = {"info": 0}
        
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Gymnasium requiere manejar seed y options
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        state = np.expand_dims(state, axis=-1)  # Añadimos dimensión de canal
        info = {}
        return state, info
    
    def getReward(self):
        return self.game.get_total_reward()

    def render(self):
        # Implementación opcional para renderizado
        pass

    def close(self):
        self.game.close()