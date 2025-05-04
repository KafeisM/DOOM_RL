# common/envs.py
import typing as t
import numpy as np
import vizdoom
import gymnasium as gym
from gymnasium import spaces, Env
from vizdoom import GameVariable
from common.frame_processor import default_frame_processor

Frame = np.ndarray

class DoomEnv(Env):
    def __init__(self,
                 game: vizdoom.DoomGame,
                 frame_processor: t.Callable = default_frame_processor,
                 frame_skip: int = 4):
        super().__init__()

        self.game = game
        self.frame_skip = frame_skip
        self.frame_processor = frame_processor

        # Acción discreta
        self.action_space = spaces.Discrete(game.get_available_buttons_size())

        # Procesar frame inicial para definir observation_space
        h, w, c = game.get_screen_height(), game.get_screen_width(), game.get_screen_channels()
        new_h, new_w, new_c = frame_processor(np.zeros((h, w, c))).shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(new_h, new_w, new_c), dtype=np.uint8)

        self.possible_actions = np.eye(self.action_space.n).tolist()
        self.empty_frame = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.state = self.empty_frame

    def step(self, action: int) -> t.Tuple[Frame, float, bool, bool, t.Dict]:
        reward = self.game.make_action(self.possible_actions[action], self.frame_skip)
        terminated = self.game.is_episode_finished()
        truncated = False  # Opcional: cortar por tiempo
        self.state = self._get_frame(terminated)

        return self.state, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None) -> t.Tuple[Frame, t.Dict]:
        self.game.new_episode()
        self.state = self._get_frame()
        return self.state, {}

    def close(self):
        self.game.close()

    def render(self, mode="human"):
        pass

    def _get_frame(self, done=False) -> Frame:
        return self.frame_processor(
            self.game.get_state().screen_buffer) if not done else self.empty_frame


def create_env(scenario: str, **kwargs) -> DoomEnv:
    game = vizdoom.DoomGame()
    game.load_config(f"scenarios/{scenario}.cfg")
    game.init()
    return DoomEnv(game, **kwargs)

