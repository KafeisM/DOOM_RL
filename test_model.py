import time
import cv2
import numpy as np
from stable_baselines3 import PPO,DQN
from common.DoomEnv import BaseVizDoomEnv
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    # Rutas del modelo y escenario
    MODEL_PATH = "DQN/train-models/train_defend_center/dqn_basic_final.zip"
    SCENARIO_PATH = "ViZDoom/scenarios/defend_the_center.cfg"

    # Cargar modelo
    try:
        model = DQN.load(MODEL_PATH)
        print(f"✓ Modelo cargado: {MODEL_PATH}")
    except Exception as e:
        print(f"✗ Error cargando modelo: {e}")
        return

    # Crear entorno con renderizado
    env = BaseVizDoomEnv(SCENARIO_PATH, num_actions=3, render=True)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Opcional: Evaluar rendimiento medio
    print("Evaluando rendimiento medio...")
    # Adaptamos evaluate_policy para la nueva interfaz de Gymnasium
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    print(f"Recompensa media: {mean_reward}")

    # Visualizar episodios
    print("\nVisualizando comportamiento del agente...")
    for episode in range(5):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0

        print(f"\n--- Episodio {episode + 1}/5 ---")

        while not done and not truncated:
            # Predecir acción
            action, _ = model.predict(obs, deterministic=True)

            # Ejecutar acción
            obs, reward, done, truncated, info = env.step(action)

            # Creamos un frame para mostrar (ya que render() devuelve None)
            # Mostramos directamente la observación
            frame = obs.squeeze()  # Eliminamos dimensión de canal si es necesario
            if frame.shape == (100, 160):  # si es grayscale
                # Convertir a RGB para visualización
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            if frame is not None:
                # Redimensionar para mejor visualización
                frame = cv2.resize(frame, (640, 400))
                cv2.imshow("Doom RL Agent", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Visualización interrumpida por el usuario")
                    break

            # Actualizar estadísticas
            total_reward += reward
            steps += 1

            # Pequeña pausa para mejor visualización
            time.sleep(0.05)

        print(f"Episodio {episode + 1}: pasos={steps}, recompensa={total_reward:.1f}")
        time.sleep(1)  # Pausa entre episodios

    # Cerrar recursos
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()