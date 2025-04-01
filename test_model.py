import time
from stable_baselines3 import PPO
from defend_center import DefendCenterEnv  # Importar el entorno correcto
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    # Cargar modelo entrenado
    model = PPO.load("train/train_defend_center/best_model_100000.zip")

    # Crear el mismo entorno que se usó para entrenar, pero con render=True
    env = DefendCenterEnv(render=True)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Opción 1: Evaluación automática (descomenta para usar)
    # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Opción 2: Prueba manual con visualización
    for episode in range(5):  # Jugar 5 episodios
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

    # Cerrar el entorno al finalizar
    env.close()

if __name__ == '__main__':
    main()