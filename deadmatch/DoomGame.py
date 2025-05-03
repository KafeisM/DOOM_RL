# doom_env.py
import os
import numpy as np
import vizdoom as vzd
import gymnasium as gym
from gymnasium import spaces

class DoomEnv(gym.Env):
    """
    Custom Gymnasium environment for ViZDoom Deathmatch.
    Uses the low-level DoomGame API and a small discrete action set.
    """
    def __init__(self,
                 config_file: str = "deathmatch.cfg",
                 frame_skip: int = 4,
                 host_window_visible: bool = False):
        super(DoomEnv, self).__init__()
        # Initialize DoomGame
        self.game = vzd.DoomGame()
        scenarios_path = vzd.scenarios_path
        self.game.load_config(os.path.join(scenarios_path, config_file))
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.set_window_visible(host_window_visible)
        self.game.init()

        # Build discrete action mapping
        self.available_buttons = self.game.get_available_buttons()
        # Define a small set of useful actions:
        combos = [
            [vzd.Button.MOVE_FORWARD],
            [vzd.Button.MOVE_LEFT],
            [vzd.Button.MOVE_RIGHT],
            [vzd.Button.TURN_LEFT],
            [vzd.Button.TURN_RIGHT],
            [vzd.Button.ATTACK],
            [vzd.Button.MOVE_FORWARD, vzd.Button.ATTACK],
            [vzd.Button.MOVE_LEFT,    vzd.Button.ATTACK],
            [vzd.Button.MOVE_RIGHT,   vzd.Button.ATTACK],
        ]
        n_btn = len(self.available_buttons)
        self._action_map = []
        for combo in combos:
            arr = [0] * n_btn
            for b in combo:
                if b in self.available_buttons:
                    idx = self.available_buttons.index(b)
                    arr[idx] = 1
            self._action_map.append(arr)

        # Gym spaces
        self.action_space = spaces.Discrete(len(self._action_map))
        h, w = self.game.get_screen_height(), self.game.get_screen_width()
        self._screen_shape = (h, w, 3)
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(0, 255, shape=self._screen_shape, dtype=np.uint8),
            "gamevariables": spaces.Box(
                -np.inf, np.inf,
                shape=(len(self.game.get_available_game_variables()),),
                dtype=np.float32
            )
        })
        self.frame_skip = frame_skip

    def reset(self, *, seed=None, options=None):
        self.game.new_episode()
        state = self.game.get_state()
        screen = state.screen_buffer
        gv     = np.array(state.game_variables, dtype=np.float32)
        obs = {"screen": screen, "gamevariables": gv}
        return obs, {}

    def step(self, action: int):
        buttons = self._action_map[action]
        reward = self.game.make_action(buttons, self.frame_skip)
        done   = self.game.is_episode_finished()
        if not done:
            state = self.game.get_state()
            screen = state.screen_buffer
            gv     = np.array(state.game_variables, dtype=np.float32)
            obs = {"screen": screen, "gamevariables": gv}
        else:
            obs = {
                "screen": np.zeros(self._screen_shape, dtype=np.uint8),
                "gamevariables": np.zeros(
                    (len(self.game.get_available_game_variables()),),
                    dtype=np.float32
                )
            }
        return obs, reward, done, False, {}

    def close(self):
        self.game.close()
