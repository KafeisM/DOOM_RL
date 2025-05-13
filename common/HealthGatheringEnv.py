from vizdoom import *
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np
import os.path

class HealthGatheringEnv(Env):
    def __init__(self, scenario_path, render=False):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config(scenario_path)
        
        # Render the game
        if render:
            self.game.set_window_visible(True)
        else:
            self.game.set_window_visible(False)
            
        self.game.init()
        
        # Estado inicial
        self.current_health = 100
        self.current_itemcount = 0  

        # Espacios de observación y acción
        self.observation_space = Box(low=0, high=255, shape=(120, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(self.game.get_available_buttons_size())  # Acciones: [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]

        print(f"Variables de juego disponibles: {self.game.get_available_game_variables()}")
        print(f"Acciones disponibles: {self.game.get_available_buttons()}")
        print(f"Formato de pantalla: {self.game.get_screen_format()}")


    def step(self, action):
        buttons = self.game.get_available_buttons_size()
        actions = np.identity(buttons)
        movement_reward = (self.game.make_action(actions[action], 4)) / 4
        reward = 0
        
        terminated = False
        truncated = False
        
        if self.game.get_state():
            # Obtener variables del juego (HEALTH y ITEMCOUNT)
            state = self.game.get_state().screen_buffer
            state = np.expand_dims(state, axis=-1)  # Añadimos dimensión de canal
            game_variables = self.game.get_state().game_variables
            health, itemcount = game_variables
            
            # --- Reward Shaping ---
            health_delta = health - self.current_health
            self.current_health = health

            itemcount_delta = itemcount - self.current_itemcount
            self.current_itemcount = itemcount
            
            # 1. Recompensa por recolectar kits médicos (+50 por ítem)
            if itemcount_delta > 0:
                item_reward += 50.0 * itemcount_delta
            
            # 2. Recompensa/penalización por cambios en salud
            if health_delta > 0:
                health_reward += 2.0 * health_delta  # +2 por punto de salud ganado
                self.steps_without_health = 0
            elif health_delta < 0:
                health_reward -= 10.0 * abs(health_delta)  # -10 por punto de daño
            
        
            self.steps_without_health += 1 if health_delta <= 0 else 0
            
            reward = movement_reward + item_reward + health_reward
            reward = reward / 1000
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)
        
        terminated = self.game.is_episode_finished()
        truncated = False

        info = {"info": 0}
            
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        state = np.expand_dims(state, axis=-1)  # Añadimos dimensión de canal
        info = {}
        return state, info
    
    def render(self):
        pass

    def close(self):
        self.game.close()