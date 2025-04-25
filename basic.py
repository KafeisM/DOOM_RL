import cv2
from stable_baselines3 import ppo
from common.callbacks import TrainAndLoggingCallback
from common import envs
from stable_baselines3.common import policies


def main():
    CHECKPOINT_DIR = 'train/train_basic'
    LOG_DIR = 'logs/log_basic'

    env_args = {
        'scenario': 'basic',
        'frame_skip': 4,
        'frame_processor': envs.default_frame_processor
    }


    env = envs.create_vec_env(**env_args)
    eval_env = envs.create_vec_env(**env_args)
    model =  ppo.PPO(policy=policies.ActorCriticCnnPolicy,
                    env=env,
                    learning_rate=1e-4,
                    tensorboard_log='logs/tensorboard')

    callback = TrainAndLoggingCallback(check_freq=10000,
                                       save_path=CHECKPOINT_DIR,
                                       eval_env=eval_env,  # Entorno para evaluación
                                       n_eval_episodes=10,  # Número de episodios para evaluar
                                       eval_freq=2500,  # Frecuencia de evaluación
                                       save_best=True  # Guardar el mejor modelo
                                       )

    model.learn(total_timesteps=25000, callback=callback)
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main()