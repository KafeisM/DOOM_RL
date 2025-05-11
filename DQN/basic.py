import cv2
import numpy as np
import vizdoom as vzd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces, Env
from common.DoomEnv import BaseVizDoomEnv
from common.callbacks import TrainAndLoggingCallback
import os


class BasicEnv(BaseVizDoomEnv):
    def __init__(self, render=False):
        super().__init__("../ViZDoom/scenarios/basic.cfg", 3, render)

# 2. Configuración del entrenamiento
def main():
    # Directorios para guardar modelos y logs
    CHECKPOINT_DIR = "train-models/trained_model"
    LOG_DIR = "../logs/DQN/log_basic"

    # Crear entorno
    env = BasicEnv()

    # Configuración de DQN optimizada para ViZDoom Basic
    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=10000,  # Tamaño del buffer de experiencia
        learning_starts=5000,  # Pasos iniciales de exploración
        batch_size=32,  # Tamaño del batch para entrenamiento
        target_update_interval=500,  # Actualizar red objetivo cada 1000 pasos
        tensorboard_log=LOG_DIR,
        verbose=1
    )

    # Callback para evaluación
    callback = TrainAndLoggingCallback(check_freq=20000, save_path=CHECKPOINT_DIR)

    # Entrenamiento extendido (DQN necesita más tiempo que PPO)
    model.learn(total_timesteps=100000,callback=callback)

    # Guardar modelo final
    model.save(os.path.join(CHECKPOINT_DIR, "dqn_basic_final"))
    env.close()


if __name__ == "__main__":
    main()