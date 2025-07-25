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
        self.steps_without_health = 0

        # Espacios de observación y acción
        self.observation_space = Box(low=0, high=255, shape=(120, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(self.game.get_available_buttons_size())  # Acciones: [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]

        print(f"Variables de juego disponibles: {self.game.get_available_game_variables()}")
        print(f"Acciones disponibles: {self.game.get_available_buttons()}")
        print(f"Formato de pantalla: {self.game.get_screen_format()}")

    def step(self, action):
        buttons = self.game.get_available_buttons_size()
        actions = np.identity(buttons)

        # Ejecutar acción
        self.game.make_action(actions[action], 4)

        # Recompensa base inicializada
        reward = 0.0

        if self.game.get_state():
            # Obtener el estado y variables del juego
            state = self.game.get_state().screen_buffer
            state = np.expand_dims(state, axis=-1)
            game_variables = self.game.get_state().game_variables
            health, itemcount = game_variables

            # Calcular cambios en variables
            health_delta = health - self.current_health
            self.current_health = health

            itemcount_delta = itemcount - self.current_itemcount
            self.current_itemcount = itemcount

            # Recompensa normalizada entre 0-1
            # Nota: La documentación menciona que hay +1 por cada tic sobrevivido
            # Esta recompensa ya la aplica el juego según la documentación

            # Normalización de recompensas adicionales
            # 1. Recompensa por recoger kit médico
            item_reward = 0.3 * itemcount_delta if itemcount_delta > 0 else 0

            # 2. Recompensa por cambios en salud
            if health_delta > 0:
                health_reward = 0.02 * health_delta
                self.steps_without_health = 0
            elif health_delta < 0:
                health_reward = -0.05 * abs(health_delta)
            else:
                health_reward = 0

            self.steps_without_health += 1 if health_delta <= 0 else 0

            # Pequeña penalización por estar mucho tiempo sin recuperar salud
            starvation_penalty = -0.001 * min(self.steps_without_health / 200,
                                              0.1) if self.steps_without_health > 100 else 0

            # Recompensa total
            reward = item_reward + health_reward + starvation_penalty

            # Limitar recompensa para asegurar rango 0-1
            reward = np.clip(reward, -1.0, 1.0)
        else:
            state = np.zeros(self.observation_space.shape, dtype=np.uint8)

        # Verificar si el episodio ha terminado
        terminated = self.game.is_episode_finished()
        truncated = False

        # La documentación menciona que hay -100 por muerte
        # Esta penalización ya la aplica el juego automáticamente

        info = {"health": self.current_health, "items": self.current_itemcount}

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