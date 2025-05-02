import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from common.DoomEnv import BaseVizDoomEnv
from common.EnvRew import VizDoomEnv
import cv2
import numpy as np
import os
from datetime import datetime

def main():
    # Cargar modelo entrenado (ajusta la ruta si es necesario)
    model_path = "models/ppo_vizdoom_curriculum_final.zip"
    try:
        model = PPO.load(model_path)
        print(f"Modelo cargado exitosamente: {model_path}")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Crear el mismo entorno que se usó para entrenar
    scenario_path = "Scenarios/deadly_corridor/deadly_corridor - t5.cfg"
    
    try:
        env = VizDoomEnv(
            scenario_path,
            render=True  # Activamos la visualización
        )
        print("Entorno creado correctamente")
    except Exception as e:
        print(f"Error al crear el entorno: {e}")
        return

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Configuración para la grabación de video
    videos_dir = "videos"
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(videos_dir, f"doom_model_test_{timestamp}.mp4")
    
    # Determinar las dimensiones del frame
    # Hacemos reset para asegurarnos que el entorno esté listo
    obs, _ = env.reset()
    test_frame = env.render()
    
    if test_frame is None:
        # Si render() no devuelve un frame, usamos la observación
        print("No se pudo obtener frame desde render(), usando observación como base")
        test_frame = obs
        if len(test_frame.shape) == 3 and test_frame.shape[2] == 1:
            # Convertir de escala de grises a BGR para el video
            test_frame = cv2.cvtColor(test_frame.squeeze(), cv2.COLOR_GRAY2BGR)
    
    # Asegurarnos que el frame tenga el formato correcto
    if len(test_frame.shape) == 2:  # Es una imagen en escala de grises
        test_frame = cv2.cvtColor(test_frame, cv2.COLOR_GRAY2BGR)
    
    height, width = test_frame.shape[:2]
    print(f"Tamaño del frame para el video: {width}x{height}")
    fps = 20  # Frames por segundo del video
    
    # Inicializar el escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    print(f"Grabando video en: {video_filename}")

    # Prueba con visualización y grabación
    for episode in range(15):  # Jugar 15 episodios
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\nEpisodio {episode+1}")
        print("-" * 20)
        
        while not done:
            # Predecir acción usando el modelo entrenado
            action, _ = model.predict(obs)
            
            # Ejecutar acción en el entorno
            obs, reward, done, truncated, info = env.step(action)
            
            # Capturar el frame actual para el video
            frame = env.render()
            
            # Si render() no devuelve un frame, usar la observación
            if frame is None:
                frame = obs
                if len(frame.shape) == 3 and frame.shape[2] == 1:
                    # Convertir de escala de grises a BGR para el video
                    frame = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2BGR)
            
            # Asegurarse de que el frame tenga el formato correcto para OpenCV (BGR)
            if frame is not None:
                if len(frame.shape) == 2:  # Frame en escala de grises
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 3 and frame.dtype == np.float32:
                    frame = (frame * 255).astype(np.uint8)  # Normalizar si es necesario
                
                # Escribir el frame en el video
                video_writer.write(frame)
            
            # Mostrar información (opcional)
            if steps % 10 == 0:  # Mostrar info cada 10 pasos
                print(f"Paso {steps}, Acción: {action}, Recompensa: {reward:.2f}")
            
            # Pausar brevemente para ver la visualización mejor
            time.sleep(0.05)
            
            # Actualizar contadores
            total_reward += reward
            steps += 1
            
            # Salir si se trunca el episodio
            if truncated:
                print("Episodio truncado")
                break
                
        print(f"Episodio {episode+1} completado:")
        print(f"Total de pasos: {steps}")
        print(f"Recompensa total: {total_reward:.2f}")
        time.sleep(1)  # Pausa entre episodios

    # Liberar recursos del video y cerrar el entorno
    video_writer.release()
    env.close()
    
    print(f"Video guardado en: {video_filename}")

if __name__ == '__main__':
    main()