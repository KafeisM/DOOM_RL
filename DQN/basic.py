import cv2
import numpy as np
import vizdoom as vzd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces, Env
import os


# 1. Entorno personalizado que hereda de gymnasium.Env
class VizDoomBasicEnv(Env):
    def __init__(self, config_path, render=False):
        super().__init__()
        self.game = vzd.DoomGame()
        self.game.load_config(config_path)
        self.game.set_window_visible(render)
        self.game.init()

        # Espacios de observación y acción
        self.observation_space = spaces.Box(low=0, high=255, shape=(120, 160, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)  # [left, right, shoot]
        self.metadata = {'render_modes': ['human']}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self._process_frame(self.game.get_state().screen_buffer)
        return state, {}

    def step(self, action):
        reward = self.game.make_action([int(action == 0), int(action == 1), int(action == 2)], 4)
        done = self.game.is_episode_finished()

        if done:
            next_state = np.zeros(self.observation_space.shape, dtype=np.uint8)
        else:
            next_state = self._process_frame(self.game.get_state().screen_buffer)

        return next_state, reward, done, False, {}

    def _process_frame(self, frame):
        gray = cv2.cvtColor(np.moveaxis(frame, 0, -1), cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_CUBIC)
        return np.expand_dims(resized, axis=-1)

    def render(self, mode='human'):
        if mode == 'human':
            return self.game.get_state().screen_buffer
        return None

    def close(self):
        self.game.close()


# 2. Configuración del entrenamiento
def main():
    # Directorios para guardar modelos y logs
    CHECKPOINT_DIR = "train-models/trained_model"
    LOG_DIR = "../logs/DQN/log_basic"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Crear entorno
    env = VizDoomBasicEnv("basic.cfg", render=False)
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    # Configuración de DQN optimizada para ViZDoom Basic
    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=100_000,  # Tamaño del buffer de experiencia
        learning_starts=10_000,  # Pasos iniciales de exploración
        batch_size=32,  # Tamaño del batch para entrenamiento
        learning_rate=1e-4,  # Tasa de aprendizaje
        gamma=0.99,  # Factor de descuento
        exploration_final_eps=0.05,  # Exploración mínima (5%)
        exploration_fraction=0.2,  # Fracción de entrenamiento para decaer epsilon
        train_freq=4,  # Frecuencia de entrenamiento
        gradient_steps=1,  # Pasos de gradiente por actualización
        target_update_interval=1000,  # Actualizar red objetivo cada 1000 pasos
        tensorboard_log=LOG_DIR,
        verbose=1
    )

    # Callback para evaluación
    eval_callback = EvalCallback(
        env,
        best_model_save_path=CHECKPOINT_DIR,
        log_path=LOG_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    # Entrenamiento extendido (DQN necesita más tiempo que PPO)
    model.learn(
        total_timesteps=200_000,
        callback=eval_callback,
        tb_log_name="dqn_basic"
    )

    # Guardar modelo final
    model.save(os.path.join(CHECKPOINT_DIR, "dqn_basic_final"))
    env.close()


if __name__ == "__main__":
    main()