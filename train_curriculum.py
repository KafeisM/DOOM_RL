from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

from common.EnvRew import VizDoomEnv
from common.shaped_reward_wrapper import ShapedRewardWrapper
import os
import gymnasium as gym

# Carpetas
LOG_DIR = "./logs/experimento3"
MODEL_DIR = "./models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    # Definición de niveles con nuevo curriculum
    levels = [
        {"path": "Scenarios/deadly_corridor/deadly_corridor - t1.cfg", "name": "nivel1", "steps": 50000},
        {"path": "Scenarios/deadly_corridor/deadly_corridor - t2.cfg", "name": "nivel2", "steps": 100000},
        {"path": "Scenarios/deadly_corridor/deadly_corridor - t3.cfg", "name": "nivel3", "steps": 200000},
        {"path": "Scenarios/deadly_corridor/deadly_corridor - t4.cfg", "name": "nivel4", "steps": 200000},
        {"path": "Scenarios/deadly_corridor/deadly_corridor - t5.cfg", "name": "nivel5", "steps": 300000},
    ]

    model = None
    for level_idx, level in enumerate(levels):
        cfg_path = level["path"]
        level_name = level["name"]
        steps = level["steps"]

        print(f"\nEntrenando nivel {level_idx} ({level_name}): {cfg_path} con {steps} pasos")

        # Factory de entorno con múltiples instancias
        def make_env():
            env = VizDoomEnv(cfg_path, render=False, frame_skip=4)
            env = ShapedRewardWrapper(env)
            return Monitor(env, os.path.join(LOG_DIR, f"monitor_{level_name}.csv"))

        env = DummyVecEnv([make_env for _ in range(4)])
        env = VecTransposeImage(env)
        env = VecFrameStack(env, n_stack=4)

        eval_env = DummyVecEnv([make_env])
        eval_env = VecTransposeImage(eval_env)
        eval_env = VecFrameStack(eval_env, n_stack=4)

        # Configurar callbacks
        save_freq = max(1000, steps // 5)
        eval_freq = max(1000, steps // 6)

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(MODEL_DIR, level_name),
            name_prefix=f"ppo_vizdoom_{level_name}"
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(MODEL_DIR, level_name),
            log_path=os.path.join(LOG_DIR, f"eval_{level_name}"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )

        # Inicialización o actualización del modelo
        try:
            if model is None:
                model = PPO(
                    "CnnPolicy",
                    env,
                    n_steps=2048,
                    batch_size=64,
                    gae_lambda=0.95,
                    gamma=0.99,
                    learning_rate=5e-5,
                    ent_coef=0.05,
                    clip_range=0.05,
                    n_epochs=5,
                    verbose=1,
                    tensorboard_log=LOG_DIR,
                    max_grad_norm=0.5,
                )
            else:
                model.set_env(env)

            print(f"🏋️ Iniciando entrenamiento de {steps} pasos...")
            model.learn(total_timesteps=steps, callback=[checkpoint_callback, eval_callback])

            # Guardar el mejor modelo de nivel
            model_path = os.path.join(MODEL_DIR, f"ppo_vizdoom_{level_name}_final.zip")
            model.save(model_path)
            print(f"💾 Modelo de {level_name} guardado en {model_path}")

        except Exception as e:
            print(f"⚠️ Error durante el entrenamiento: {e}")
        finally:
            env.close()
            eval_env.close()

    # Guardar modelo final completo
    if model:
        final_path = os.path.join(MODEL_DIR, "ppo_vizdoom_curriculum_exp003.zip")
        model.save(final_path)
        print(f"\n✅ Entrenamiento completo guardado en {final_path}")


if __name__ == "__main__":
    main()
