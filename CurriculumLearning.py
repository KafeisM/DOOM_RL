from stable_baselines3 import PPO
from common.callbacks import TrainAndLoggingCallback
from common.DoomEnvRew import VizDoomGymReward
import os
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor


def make_env(config_path):
    """
    Función para crear el entorno con la configuración especificada
    """

    def _init():
        env = VizDoomGymReward(config=config_path)
        env = Monitor(env)
        return env

    return _init


def main():
    CHECKPOINT_DIR = 'train/train_deadly_corridor_curriculum'
    LOG_DIR = 'logs/log_deadly_corridor_curriculum'

    # Crear directorios si no existen
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Parámetros optimizados para el entrenamiento
    model_params = {
        'policy': 'CnnPolicy',
        'tensorboard_log': LOG_DIR,
        'verbose': 1,
        'learning_rate': 0.00005,  # Aumentado para mejor adaptación
        'n_steps': 12288,  # Más pasos por actualización
        'clip_range': 0.2,  # Mayor flexibilidad
        'gamma': 0.99,  # Visión a largo plazo
        'gae_lambda': 0.95,  # Balance adecuado
        'ent_coef': 0.01,  # Fomentar exploración
        'vf_coef': 0.5  # Balance política/valor
    }

    # Definir todas las etapas del curriculum
    curriculum_stages = [
        './Scenarios/deadly_corridor/deadly_corridor - t1.cfg',
        './Scenarios/deadly_corridor/deadly_corridor - t2.cfg',
        './Scenarios/deadly_corridor/deadly_corridor - t3.cfg',
        './Scenarios/deadly_corridor/deadly_corridor - t4.cfg',
        './Scenarios/deadly_corridor/deadly_corridor - t5.cfg'
    ]

    # Pasos para cada etapa (incremento progresivo)
    timesteps_per_stage = [200000, 200000, 300000, 300000, 400000]

    # Buscar si hay un checkpoint previo para continuar
    last_checkpoint = None
    last_stage = 0

    for i in range(len(curriculum_stages), 0, -1):
        model_path = f"{CHECKPOINT_DIR}/best_model_stage{i}.zip"
        if os.path.exists(model_path):
            last_checkpoint = model_path
            last_stage = i
            break

    # Modelo inicial (nuevo o cargado)
    if last_checkpoint:
        print(f"Cargando modelo desde {last_checkpoint}")
        model = PPO.load(last_checkpoint)
        start_stage = last_stage
    else:
        print("Iniciando entrenamiento desde cero")
        # Crear y vectorizar el entorno para la primera etapa
        env = DummyVecEnv([make_env(curriculum_stages[0])])
        env = VecTransposeImage(env)

        # Crear modelo nuevo
        model = PPO(**model_params, env=env)
        start_stage = 1

    # Entrenar en cada etapa del curriculum
    for i in range(start_stage - 1, len(curriculum_stages)):
        stage_num = i + 1
        config_path = curriculum_stages[i]

        print(f"\n{'=' * 50}")
        print(f"Entrenando etapa {stage_num}: {config_path}")
        print(f"Pasos: {timesteps_per_stage[i]}")
        print(f"{'=' * 50}\n")

        # Crear entorno para esta etapa
        env = DummyVecEnv([make_env(config_path)])
        env = VecTransposeImage(env)

        # Actualizar entorno del modelo
        model.set_env(env)

        # Configurar callback para guardar checkpoints
        stage_dir = f"{CHECKPOINT_DIR}/stage{stage_num}"
        os.makedirs(stage_dir, exist_ok=True)
        callback = TrainAndLoggingCallback(check_freq=20000, save_path=stage_dir)

        # Entrenar modelo en esta etapa
        model.learn(total_timesteps=timesteps_per_stage[i], callback=callback)

        # Guardar mejor modelo de esta etapa
        model_stage_path = f"{CHECKPOINT_DIR}/best_model_stage{stage_num}.zip"
        model.save(model_stage_path)
        print(f"Modelo guardado como {model_stage_path}")

        # Cerrar entorno
        env.close()

    print("\n¡Entrenamiento por curriculum completado!")
    print(f"Todos los modelos guardados en {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()