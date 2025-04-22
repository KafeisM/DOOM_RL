#import PPO for training
from stable_baselines3 import PPO
from common.callbacks import TrainAndLoggingCallback

from common.DoomEnvRew import BaseVizDoomEnvPrueba  # Importar la clase base

class DefendCenterEnv(BaseVizDoomEnvPrueba):
    def __init__(self, render=False):
        # Configuración específica para este escenario
        game_variables_config = {
            "variables": ["ammo", "health"],  # Variables que nos interesan
            "weights": [5.0, 10.0],           # Pesos para el reward shaping
        }
        
        super().__init__(
            "./ViZDoom/scenarios/defend_the_center.cfg", 
            3, 
            render,
            game_variables_config
        )

def main():
    CHECKPOINT_DIR = 'train/train_defend_center_2'
    LOG_DIR = 'logs/log_defend_center'

    callback = TrainAndLoggingCallback(check_freq=20000, save_path=CHECKPOINT_DIR)

    env = DefendCenterEnv()
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)
    model.learn(total_timesteps=100000, callback=callback)


if __name__ == "__main__":
    main()