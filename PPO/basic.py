#import PPO for training
from stable_baselines3 import PPO
from common.DoomEnv import BaseVizDoomEnv  # Importar la clase específica
from common.callbacks import TrainAndLoggingCallback


class BasicEnv(BaseVizDoomEnv):
    def __init__(self, render=False):
        super().__init__("./ViZDoom/scenarios/basic.cfg", 3, render)



def main():
    CHECKPOINT_DIR = 'train/train_basic'
    LOG_DIR = 'logs/log_basic'

    callback = TrainAndLoggingCallback(check_freq=20000, save_path=CHECKPOINT_DIR)

    env = BasicEnv()
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048)
    model.learn(total_timesteps=100000, callback=callback)


if __name__ == "__main__":
    main()