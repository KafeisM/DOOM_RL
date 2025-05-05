import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class TrainAndLoggingCallback(BaseCallback):
    """
    Callback personalizado para:
    - Guardar modelos periódicamente
    - Registrar métricas extensivas
    - Evaluación periódica del modelo
    """
    
    def __init__(self, check_freq, save_path, eval_freq=None, eval_env=None, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.best_mean_reward = -np.inf

    def _init_callback(self):
        # Crear directorios si no existen
        os.makedirs(self.save_path, exist_ok=True)
        if self.eval_env is not None:
            os.makedirs(os.path.join(self.save_path, 'best_model'), exist_ok=True)

    def _on_step(self) -> bool:
        # Guardar modelo periódicamente
        if self.n_calls % self.check_freq == 0:
            path = os.path.join(self.save_path, f'model_{self.num_timesteps}_steps')
            self.model.save(path)
            
            # Registrar métricas de entrenamiento
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer if 'r' in ep_info]
                if rewards:
                    self.logger.record('train/mean_reward', np.mean(rewards))
                    self.logger.record('train/max_reward', np.max(rewards))
                    self.logger.record('train/min_reward', np.min(rewards))
                    self.logger.record('train/std_reward', np.std(rewards))
            
            # Evaluación periódica
            if self.eval_freq and self.n_calls % self.eval_freq == 0 and self.eval_env:
                mean_reward, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=5)
                self.logger.record('eval/mean_reward', mean_reward)
                
                # Guardar mejor modelo
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(os.path.join(self.save_path, 'best_model/best_model'))
            
            self.logger.dump(self.num_timesteps)
            
        return True