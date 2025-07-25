# src/algorithms/ppo.py

import argparse
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from common.HealthGatheringEnv import HealthGatheringEnv
from common.callbacks import TrainAndLoggingCallback

CFG_PATH = "../Scenarios/health_gathering/health_gathering.cfg"
SAVE_PATH = "./train - models/train_health"
LOG_PATH = "../logs/PPO/log_health"
TIMESTEPS = 1_000_000


def setup_env(config_path):
    env = HealthGatheringEnv(config_path, render=False)
    env = Monitor(env)
    return env


def train_ppo():
    # 1) Crea el entorno

    env = setup_env(CFG_PATH)

    eval_env = HealthGatheringEnv(scenario_path=CFG_PATH)  # Entorno independiente para evaluación

    # 2) Define un callback para salvar checkpoints cada X pasos
    callbacks = TrainAndLoggingCallback(
        check_freq=100_000,
        save_path=SAVE_PATH,
        eval_freq=50_000,
        eval_env=eval_env
    )

    # 3) Instancia el modelo PPO con red convolucional
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_PATH, verbose=1, learning_rate=0.0001, n_steps=2048)

    model.set_logger(configure(LOG_PATH, ["stdout", "tensorboard", "csv"]))

    # 4) Entrena
    model.learn(total_timesteps=TIMESTEPS, callback=callbacks)

    # 5) Guarda el modelo final
    model.save(f"{SAVE_PATH}/ppo_final")

    env.close()


if __name__ == "__main__":
    train_ppo()
