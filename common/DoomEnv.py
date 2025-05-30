from vizdoom import DoomGame, ScreenResolution, ScreenFormat
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np

class BaseVizDoomEnv(gym.Env):
    """Clase base para entornos VizDoom simples con semilla antes de init."""
    def __init__(self, scenario_path, num_actions, render=False, seed=None):
        super().__init__()
        print("Loading scenario:", scenario_path)

        # Configura el juego
        self.game = DoomGame()

        # Aplica semilla antes de cargar la configuración y de init
        if seed is not None:
            print(f"[DEBUG] Applying seed before load_config/init: {seed}")
            self.game.set_seed(seed)
            print(f"[SEED] DoomGame seed set to {seed} (pre-load)")

        # Carga configuración del escenario
        self.game.load_config(scenario_path)

        # Configuraciones de pantalla
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_window_visible(render)

        # Inicializa el motor con la semilla aplicada
        print("[DEBUG] Calling game.init() with current seed")
        self.game.init()

        # Reaplica semilla tras init para asegurar consistencia
        if seed is not None:
            self.game.set_seed(seed)
            print(f"[SEED] DoomGame seed re-applied to {seed} (post-init)")

        # Espacios de Gym
        self.num_actions = num_actions if num_actions > 0 else self.game.get_available_buttons_size()
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
            state = np.expand_dims(state, axis=-1)
            ammo = self.game.get_state().game_variables[0]
        else:
            state = np.zeros(self.observation_space.shape)
            ammo = 0
        info = {"ammo": ammo}
        terminated = self.game.is_episode_finished()
        truncated = False
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reinicia episodio
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        state = np.expand_dims(state, axis=-1)
        ammo = self.game.get_state().game_variables[0]
        return state, {"ammo": ammo}

    def render(self):
        pass

    def close(self):
        self.game.close()