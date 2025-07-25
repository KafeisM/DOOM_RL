# DQN/deadly_corridor.py

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3 import DQN
from common.DeadlyCorridorEnv import VizDoomReward
from common.callbacks import TrainAndLoggingCallback

CFG_PATH = "../Scenarios/deadly_corridor/deadly_corridor.cfg"
SAVE_PATH = "train-models/deadly_corridor"
LOG_PATH = "../logs/DQN/log_deadly_corridor"
TIMESTEPS = 1_000_000


def setup_env(config_path):
    env = VizDoomReward(config_path, render=False)
    env = Monitor(env)
    return env


def train_dqn():
    # 1) Crea el entorno
    env = setup_env(CFG_PATH)
    eval_env = VizDoomReward(scenario_path=CFG_PATH)  # Entorno independiente para evaluación

    # 2) Define un callback para salvar checkpoints
    callbacks = TrainAndLoggingCallback(
        check_freq=100_000,
        save_path=SAVE_PATH,
        eval_freq=50_000,
        eval_env=eval_env
    )

    # 3) Instancia el modelo DQN con hiperparámetros apropiados
    model = DQN(
        'CnnPolicy',
        env,
        tensorboard_log=LOG_PATH,
        verbose=1,
        buffer_size=100000,  # Tamaño del buffer de experiencia
        learning_starts=50000,  # Pasos antes de empezar a entrenar
        batch_size=32,  # Tamaño del lote
    )

    model.set_logger(configure(LOG_PATH, ["stdout", "tensorboard", "csv"]))

    # 4) Entrena
    model.learn(total_timesteps=TIMESTEPS, callback=callbacks)

    # 5) Guarda el modelo final
    model.save(f"{SAVE_PATH}/dqn_final")

    env.close()


if __name__ == "__main__":
    train_dqn()