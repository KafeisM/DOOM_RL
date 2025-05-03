import torch
import numpy as np
import os
import argparse
from DoomGame import DoomEnv
from env_wrappers import PreprocessFrame, FrameStack, RewardShapingWrapper
from model import DQN


def evaluate(model_path, episodes=5, render=True):
    # Crear el entorno igual que en entrenamiento
    env = DoomEnv(config_file="deathmatch.cfg", frame_skip=4, host_window_visible=render)
    env = RewardShapingWrapper(env)
    env = PreprocessFrame(env, width=108, height=60, grayscale=False)
    env = FrameStack(env, n_frames=4)

    obs_shape = env.observation_space.shape
    input_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    num_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(input_shape, num_actions).to(device)

    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Verificar si es un checkpoint completo o solo el estado del modelo
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        policy_net.load_state_dict(checkpoint['model_state'])
        print(f"Cargado checkpoint del episodio {checkpoint.get('episode', 'desconocido')}")
        print(f"Mejor recompensa: {checkpoint.get('best_reward', 'desconocida')}")
    else:
        # Para compatibilidad con formato antiguo
        policy_net.load_state_dict(checkpoint)
        print("Cargado modelo en formato antiguo")

    policy_net.eval()

    total_reward = 0.0
    total_frags = 0

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        start_kills = 0
        done = False
        ep_reward = 0.0

        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        # Obtener estadísticas finales
        if isinstance(obs, dict) and 'gamevariables' in obs:
            frags = int(obs['gamevariables'][4]) - start_kills
        else:
            frags = 0

        total_reward += ep_reward
        total_frags += frags
        print(f"Episodio {ep} | Recompensa: {ep_reward:.2f} | Frags: {frags}")

    avg_reward = total_reward / episodes
    avg_frags = total_frags / episodes
    print(f"\nPromedio en {episodes} episodios: Recompensa = {avg_reward:.2f}, Frags = {avg_frags:.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa un modelo entrenado")
    parser.add_argument("model_path", type=str,
                        help="Ruta al modelo guardado (checkpoint)")
    parser.add_argument("--episodes", "-e", type=int, default=5,
                        help="Número de episodios a ejecutar")
    parser.add_argument("--no-render", action="store_true",
                        help="Desactivar visualización")

    args = parser.parse_args()
    evaluate(args.model_path, args.episodes, not args.no_render)