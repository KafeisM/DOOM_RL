from stable_baselines3 import PPO
from common.callbacks import TrainAndLoggingCallback
from common.DoomEnvRew import VizDoomGymReward

def main():
    CHECKPOINT_DIR = 'train/train_deadly_corridor2.0'
    LOG_DIR = 'logs/log_deadly_corridor'

    callback = TrainAndLoggingCallback(check_freq=20000, save_path=CHECKPOINT_DIR)

    env = VizDoomGymReward()

    # Configuración de PPO adaptada al entorno más complejo
    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.00001, n_steps=8192,
                clip_range=.1, gamma=.95, gae_lambda=.9)

    # Entrenamiento por más tiempo debido a la complejidad del escenario
    model.learn(total_timesteps=400000, callback=callback)


if __name__ == "__main__":
    main()