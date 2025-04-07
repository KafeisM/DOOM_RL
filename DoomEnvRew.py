import vizdoom as vzd
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import cv2
import numpy as np

class BaseVizDoomEnvPrueba(gym.Env):
    """Clase base para entornos VizDoom con soporte para reward shaping"""
    
    def __init__(self, scenario_path, num_actions, render=False, game_variables_config=None):
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

        # Configuración para reward shaping
        self.game_variables_config = game_variables_config or {"variables": [], "weights": []}
        self.variable_values = {}  # Para almacenar valores anteriores de las variables
        
        # Set the Game scenario
        self.game.init()

        # Create the action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8)
        self.action_space = Discrete(num_actions)

        # Inicializar las variables de juego con sus valores iniciales
        self._init_game_variables()

    def _init_game_variables(self):
        """Inicializa el diccionario de variables de juego con sus valores actuales"""
        if self.game.get_state():
            game_vars = self.game.get_state().game_variables
            for i, var_name in enumerate(self.game_variables_config["variables"]):
                # Asumimos que el orden de las variables en game_variables coincide con el orden en variables_config
                if i < len(game_vars):
                    self.variable_values[var_name] = game_vars[i]

    def step(self, action):
        actions = np.identity(self.num_actions, dtype=int)
        movement_reward = self.game.make_action(actions[action], 4)
        
        reward = movement_reward
        info = {}
        
        
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            
            # Reward shaping basado en las variables del juego
            game_vars = self.game.get_state().game_variables
            
            # Calcular recompensas basadas en los cambios de las variables
            for i, var_name in enumerate(self.game_variables_config["variables"]):
                if i < len(game_vars):
                    current_value = game_vars[i]
                    if var_name in self.variable_values:
                        # Calcular el delta y aplicar el peso correspondiente
                        delta = current_value - self.variable_values[var_name]
                        weight = self.game_variables_config["weights"][i]
                        reward += delta * weight
                    
                    # Actualizar el valor de la variable
                    self.variable_values[var_name] = current_value
                    info[var_name] = current_value
        else:
            state = np.zeros(self.observation_space.shape)
            
        done = self.game.is_episode_finished()
        truncated = False

        return state, reward, done, truncated, info

    def render(self):
        pass

    def reset(self, seed=None):
        if seed is not None:
            super().reset(seed=seed)
            
        self.game.new_episode()
        
        info = {}
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            
            # Reinicializar las variables del juego
            self._init_game_variables()
            
            # Incluir las variables en la información de retorno
            for var_name, value in self.variable_values.items():
                info[var_name] = value
        else:
            state = np.zeros(self.observation_space.shape)
            
        return self.grayscale(state), info

    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state

    def close(self):
        self.game.close()