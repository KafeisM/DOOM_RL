import time
import cv2
import numpy as np
from stable_baselines3 import PPO
from common.DeadlyCorridorEnv import VizDoomReward
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    # Rutas del modelo y escenario
    MODEL_PATH = "PPO/trained_models/train_deadly_corridor/ppo_final.zip"
    SCENARIO_PATH = "Scenarios/deadly_corridor/deadly_corridor.cfg"

    # Cargar modelo
    try:
        model = PPO.load(MODEL_PATH)
        print(f"✓ Modelo cargado: {MODEL_PATH}")
    except Exception as e:
        print(f"✗ Error cargando modelo: {e}")
        return

    # Crear entorno con renderizado
    env = VizDoomReward(scenario_path=SCENARIO_PATH, visible=True)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Opcional: Evaluar rendimiento medio
    print("Evaluando rendimiento medio...")
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=3)
    print(f"Recompensa media: {mean_reward}")

    # Visualizar episodios
    print("\nVisualizando comportamiento del agente...")
    for episode in range(3):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0

        print(f"\n--- Episodio {episode + 1}/3 ---")

        while not done and not truncated:
            # Predecir acción
            action, _ = model.predict(obs, deterministic=True)

            # Ejecutar acción
            obs, reward, done, truncated, info = env.step(action)

            # Mostramos directamente la observación
            frame = obs.squeeze()  # Eliminamos dimensión de canal si es necesario

            # Aseguramos que el frame sea visualizable
            if frame is not None:
                # Normalizar para visualización
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)

                # Redimensionar para mejor visualización
                frame = cv2.resize(frame, (640, 480))
                # Aplicar colormap para mejor visualización
                frame_color = cv2.applyColorMap(frame, cv2.COLOR_BGR2RGB)

                cv2.imshow("Doom Deadly Corridor Agent", frame_color)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Visualización interrumpida por el usuario")
                    break

            # Actualizar estadísticas
            total_reward += reward
            steps += 1

            # Pequeña pausa para mejor visualización
            time.sleep(0.05)

        print(f"Episodio {episode + 1}: pasos={steps}, recompensa={total_reward:.2f}")
        time.sleep(1)  # Pausa entre episodios

    # Cerrar recursos
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()