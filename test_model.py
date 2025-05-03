import time
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from common.EnvRew import VizDoomEnv


def main():
    # 1) Carga del modelo
    model_path = "models/ppo_vizdoom_nivel2_final.zip"
    try:
        model = PPO.load(model_path)
        print(f"Modelo cargado exitosamente: {model_path}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # 2) Configurar entornos (igual que en entrenamiento)
    scenario_path = "Scenarios/deadly_corridor/deadly_corridor - t2.cfg"

    # Entorno para visualización directa
    viz_env = VizDoomEnv(scenario_path, render=True, frame_skip=4)

    # Entorno vectorizado con wrappers para predicción
    def make_env():
        return VizDoomEnv(scenario_path, render=False, frame_skip=4)

    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # 3) Prueba del modelo
    for episode in range(5):
        obs = env.reset()
        viz_env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"\n--- Episodio {episode + 1} ---")

        while not done:
            # Predecir acción usando el modelo entrenado
            action, _ = model.predict(obs, deterministic=True)

            # Ejecutar acción en ambos entornos
            obs, reward, done, info = env.step(action)
            viz_env.step(action[0])

            # Visualización
            img = viz_env.render()
            if img is not None:
                cv2.imshow("Doom RL", img)
                cv2.waitKey(1)

            # Actualizar estado
            total_reward += reward[0]
            steps += 1

            # Pequeña pausa para mejor visualización
            time.sleep(0.05)

            # Terminar si el episodio acaba
            if done[0]:
                break

        print(f"Episodio {episode + 1}: pasos={steps}, reward={total_reward:.1f}")

    # Cerrar recursos
    env.close()
    viz_env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()