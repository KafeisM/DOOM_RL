# Imports necesarios
from stable_baselines3 import PPO
from common.callbacks import TrainAndLoggingCallback
from common.prueba import BaseVizDoomEnvPrueba  # Importar la clase base de prueba.py


class DeadlyCorridorEnv(BaseVizDoomEnvPrueba):
    def __init__(self, render=False):
        # Configuración específica para el escenario Deadly Corridor
        game_variables_config = {
            "variables": ["health", "damage_taken", "hitcount", "ammo"],
            "weights": [1.0, -5.0, 10.0, 2.0]  # Recompensas por salud, daño (negativo), aciertos y munición
        }

        super().__init__(
            "./Scenarios/deadly_corridor/deadly_corridor - t1.cfg",
            7,  # 7 acciones disponibles según la configuración
            render,
            game_variables_config
        )


def main():
    CHECKPOINT_DIR = 'train/train_deadly_corridor'
    LOG_DIR = 'logs/log_deadly_corridor'

    callback = TrainAndLoggingCallback(check_freq=20000, save_path=CHECKPOINT_DIR)

    env = DeadlyCorridorEnv()

    # Configuración de PPO adaptada al entorno más complejo
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, n_steps=8192,
                clip_range=.1, gamma=.95, gae_lambda=.9)

    # Entrenamiento por más tiempo debido a la complejidad del escenario
    model.learn(total_timesteps=400000, callback=callback)


if __name__ == "__main__":
    main()