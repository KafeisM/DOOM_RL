import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os

# 1. Parámetros
ENV_ID = "VizdoomDefendCenter-v0"
TOTAL_TIMESTEPS = 100_000
LOG_DIR = "./ppo_vizdoom_logs"
MODEL_DIR = "./ppo_vizdoom_model"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 2. Crear entorno
# Opcional: puedes ajustar frame_skip con gym.make(..., frame_skip=4)
env = gym.make(ENV_ID, render_mode=None)
env = Monitor(env, LOG_DIR)

# 3. Crear modelo PPO con política CNN
model = PPO(
    policy="CnnPolicy",
    env=env,
    verbose=1,
    tensorboard_log=LOG_DIR
)

# 4. Callbacks (guardar modelo cada 10k steps)
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=MODEL_DIR,
    name_prefix="ppo_vizdoom"
)

# 5. Entrenamiento
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback
)

# 6. Guardar modelo final
model.save(os.path.join(MODEL_DIR, "ppo_vizdoom_final"))

env.close()
