from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from common.EnvRew import VizDoomEnv
from common.shaped_reward_wrapper import ShapedRewardWrapper
import os
import gymnasium as gym

# Carpetas
LOG_DIR = "./logs"
MODEL_DIR = "./models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    # Definición de niveles - Nombres sin espacios para evitar problemas
    levels = [
        {
            "path": "Scenarios/deadly_corridor/deadly_corridor - t1.cfg",
            "name": "nivel1",
            "steps": 30000
        },
        {
            "path": "Scenarios/deadly_corridor/deadly_corridor - t2.cfg",
            "name": "nivel2",
            "steps": 40000
        },
        {
            "path": "Scenarios/deadly_corridor/deadly_corridor - t3.cfg",
            "name": "nivel3",
            "steps": 50000
        },
        {
            "path": "Scenarios/deadly_corridor/deadly_corridor - t4.cfg",
            "name": "nivel4",
            "steps": 60000
        },
        {
            "path": "Scenarios/deadly_corridor/deadly_corridor - t5.cfg",
            "name": "nivel5",
            "steps": 70000
        }
    ]

    # Verificar que los archivos existan
    for level in levels:
        if not os.path.exists(level["path"]):
            print(f"⚠️ ADVERTENCIA: El archivo {level['path']} no existe.")

    model = None
    for level_idx, level in enumerate(levels):
        cfg_path = level["path"]
        level_name = level["name"]
        steps = level["steps"]

        print(f"\nEntrenando nivel {level_idx} ({level_name}): {cfg_path}")

        try:
            # Crear entorno de entrenamiento con wrapper vectorizado
            def make_env():
                env = VizDoomEnv(cfg_path, render=False)
                env = ShapedRewardWrapper(env)
                env = Monitor(env, filename=os.path.join(LOG_DIR, f"monitor_{level_name}"))
                return env

            # Asegurando que ambos entornos sean del mismo tipo
            env = DummyVecEnv([make_env])
            env = VecTransposeImage(env)

            # Crear entorno de evaluación con el mismo tipo
            def make_eval_env():
                env = VizDoomEnv(cfg_path, render=False)
                env = ShapedRewardWrapper(env)
                env = Monitor(env, filename=os.path.join(LOG_DIR, f"eval_monitor_{level_name}"))
                return env

            eval_env = DummyVecEnv([make_eval_env])
            eval_env = VecTransposeImage(eval_env)

        except Exception as e:
            print(f"⚠️ Error al crear el entorno: {e}")
            continue

        # Calcular frecuencias de guardado
        save_freq = max(1000, steps // 5)  # 5 checkpoints por nivel
        eval_freq = max(1000, steps // 6)  # 6 evaluaciones por nivel

        # Definir callbacks
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

        try:
            if model is None:
                model = PPO(
                    policy="CnnPolicy",
                    env=env,
                    n_steps=2048,
                    batch_size=64,
                    gae_lambda=0.95,
                    gamma=0.99,
                    n_epochs=10,
                    ent_coef=0.01,
                    learning_rate=2.5e-4,
                    clip_range=0.1,
                    verbose=1,
                    tensorboard_log=os.path.join(LOG_DIR, f"tensorboard_{level_name}"),
                )
            else:
                model.set_env(env)

            print(f"🏋️ Iniciando entrenamiento de {steps} pasos...")
            model.learn(total_timesteps=steps, callback=[checkpoint_callback, eval_callback])

            # Guardar modelo de nivel
            model_path = os.path.join(MODEL_DIR, f"ppo_vizdoom_{level_name}_final.zip")
            model.save(model_path)
            print(f"💾 Modelo guardado en {model_path}")

        except Exception as e:
            print(f"⚠️ Error durante el entrenamiento: {e}")
        finally:
            env.close()
            eval_env.close()

    # Guardar modelo final completo
    try:
        if model is not None:
            model.save(os.path.join(MODEL_DIR, "ppo_vizdoom_curriculum_final.zip"))
            print("\n✅ Entrenamiento completo guardado.")
    except Exception as e:
        print(f"⚠️ Error al guardar el modelo final: {e}")


if __name__ == "__main__":
    main()