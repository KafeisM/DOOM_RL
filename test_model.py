import time
import cv2
import numpy as np
from stable_baselines3 import PPO, DQN
from common.DoomEnv import BaseVizDoomEnv
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    # Configuración directa en el código
    model_path = "PPO/train - models/train_defend_center/final_model.zip"
    scenario = "ViZDoom/scenarios/defend_the_center.cfg"
    algorithm = "PPO"  # Opciones: "PPO" o "DQN"
    num_actions = 3
    episodes = 5

    # Cargar modelo
    try:
        if algorithm == "PPO":
            model = PPO.load(model_path)
        else:
            model = DQN.load(model_path)
        print(f"✓ Modelo cargado: {model_path}")
    except Exception as e:
        print(f"✗ Error cargando modelo: {e}")
        return

    # Crear entorno con renderizado
    try:
        env = BaseVizDoomEnv(scenario, num_actions=num_actions,
                             render=False)  # Cambiado a False para usar nuestra visualización
        print(f"✓ Entorno creado con escenario: {scenario}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
    except Exception as e:
        print(f"✗ Error creando entorno: {e}")
        return


    # Visualizar episodios
    print("\nVisualizando comportamiento del agente...")
    for episode in range(episodes):
        try:
            obs, info = env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0

            print(f"\n--- Episodio {episode + 1}/{episodes} ---")

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

                    # Mostrar frame
                    cv2.imshow("Doom Agent Visualization", frame)
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
        except Exception as e:
            print(f"✗ Error en episodio {episode + 1}: {e}")
            break

    # Cerrar recursos
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()