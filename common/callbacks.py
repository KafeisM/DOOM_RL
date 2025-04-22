import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, eval_env=None, n_eval_episodes=10,
                 eval_freq=2500, save_best=True, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

        # Parámetros para la evaluación
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.save_best = save_best

        # Crear el directorio para el mejor modelo si se especifica
        if save_best and save_path is not None:
            self.best_model_path = os.path.join(save_path, 'best_model')
            os.makedirs(self.best_model_path, exist_ok=True)

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # Guardar modelo periódicamente según check_freq
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'model_{self.n_calls}')
            self.model.save(model_path)

        # Evaluar y guardar el mejor modelo según el rendimiento
        if self.eval_env is not None and self.n_calls % self.eval_freq == 0:
            # Sincronizar normalización de entornos si aplica
            if isinstance(self.eval_env, VecEnv):
                sync_envs_normalization(self.training_env, self.eval_env)

            # Realizar la evaluación
            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0.0

                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    episode_reward += reward

                episode_rewards.append(episode_reward)

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)

            # Guardar el mejor modelo si se supera el récord anterior
            if mean_reward > self.best_mean_reward and self.save_best:
                self.best_mean_reward = mean_reward
                best_model_path = os.path.join(self.best_model_path, 'best_model')
                self.model.save(best_model_path)

                if self.verbose > 0:
                    print(f"Nuevo mejor modelo guardado en: {best_model_path}")
                    print(f"Recompensa media: {mean_reward:.2f} +/- {std_reward:.2f}")

            if self.verbose > 0:
                print(f"Evaluación en paso {self.n_calls}: {mean_reward:.2f} +/- {std_reward:.2f}")

        return True