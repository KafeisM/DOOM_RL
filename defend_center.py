#import PPO for training
from stable_baselines3 import PPO
from DoomEnv import BaseVizDoomEnv  # Importar la clase específica
from callbacks import TrainAndLoggingCallback

class DefendCenterEnv(BaseVizDoomEnv):
    def __init__(self, render=False):
        super().__init__("./ViZDoom/scenarios/defend_the_center.cfg", 3, render)



def main():
    CHECKPOINT_DIR = 'train/train_defend_center'
    LOG_DIR = 'logs/log_defend_center'

    callback = TrainAndLoggingCallback(check_freq=20000, save_path=CHECKPOINT_DIR)

    env = DefendCenterEnv()
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)
    model.learn(total_timesteps=20000, callback=callback)


if __name__ == "__main__":
    main()