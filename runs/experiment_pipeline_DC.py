"""
experiment_pipeline.py

Script para:
1. Ejecutar 50 entrenamientos independientes de PPO o DQN en un escenario VizDoom.
2. Generar CSVs con recompensas por episodio (usando Monitor).
3. Analizar los resultados: curvas de aprendizaje (mediana y cuartiles) y boxplots comparativos.
4. Realizar un test estadístico Mann–Whitney U sobre el rendimiento final.

Uso:
  - Para ejecutar los entrenamientos:
      python experiment_pipeline.py --run --algorithm PPO
      python experiment_pipeline.py --run --algorithm DQN

  - Para analizar y generar gráficas:
      python experiment_pipeline.py --analyze
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import argparse

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy.stats import mannwhitneyu
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from common.DoomEnv import BaseVizDoomEnv


def set_global_seed(env, seed: int):
    """Fija semillas globales antes de inicializar el juego y SB3."""
    # 1) Python, NumPy y PyTorch
    set_random_seed(seed)
    # 2) Semilla en DoomGame antes de init
    # La semilla ya se aplicará durante la construcción del entorno
    # 3) Espacios Gym (action/observation)
    # Se aplican en el constructor del entorno


def run_experiments(algorithm: str, n_seeds: int, total_timesteps: int,
                    scenario_cfg: str, num_actions: int, render: bool):
    """Ejecuta n_seeds entrenamientos y guarda CSVs de recompensas."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] CUDA Available: {torch.cuda.is_available()}, using device: {device}")
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    for seed in range(n_seeds):
        log_dir = f"logs/{algorithm}/seed_{seed}"
        monitor_path = os.path.join(log_dir, "monitor.csv")

        if os.path.exists(monitor_path):
            print(f"[SKIP] {algorithm} seed {seed} ya completada.")
            continue

        print(f"[RUN] {algorithm} seed={seed}")
        os.makedirs(log_dir, exist_ok=True)

        # Construir entorno con semilla antes de init
        env = BaseVizDoomEnv(scenario_cfg, num_actions, render, seed)
        env = Monitor(env, filename=monitor_path)

        # Instanciar modelo en dispositivo
        if algorithm == "PPO":
            model = PPO(
                'CnnPolicy', env,
                tensorboard_log=log_dir,
                seed=seed,
                verbose=0,
                learning_rate=0.0001,
                n_steps=4096,
                device=device
            )
        elif algorithm == "DQN":
            model = DQN(
                'CnnPolicy', env,
                tensorboard_log=log_dir,
                seed=seed,
                verbose=0,
                buffer_size=100000,
                learning_starts=50000,
                batch_size=32,
                device=device
            )
        else:
            raise ValueError(f"Algoritmo no soportado: {algorithm}")

        model.learn(total_timesteps=total_timesteps)
        # Guardar modelo final
        model.save(os.path.join(log_dir, 'final_model.zip'))
        env.close()


def load_data(algorithm: str, n_seeds: int) -> pd.DataFrame:
    runs = []
    for seed in range(n_seeds):
        path = f"logs/{algorithm}/seed_{seed}/monitor.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, skiprows=1)[['l', 'r']].copy()
        df['seed'] = seed
        runs.append(df)
    if not runs:
        raise FileNotFoundError(f"No se encontraron datos para {algorithm}")
    return pd.concat(runs, ignore_index=True)


def plot_learning_curve(ax, stats: pd.DataFrame, label: str):
    ax.plot(stats['step_bin'], stats['median'], label=f"{label} median")
    ax.fill_between(stats['step_bin'], stats['q1'], stats['q3'], alpha=0.3, label=f"{label} Q1–Q3")


def analyze_and_plot(n_seeds: int, bin_size: int, output_prefix: str):
    algorithms = ['PPO', 'DQN']
    boxplot_data = {}
    fig, ax = plt.subplots()
    for algo in algorithms:
        df = load_data(algo, n_seeds)
        df['step_bin'] = (df['l'] // bin_size) * bin_size
        stats = df.groupby('step_bin')['r'].agg(
            median='median',
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75)
        ).reset_index()
        plot_learning_curve(ax, stats, algo)

        final_steps = df['l'].max()
        final = df[df['l'] > final_steps - bin_size]
        boxplot_data[algo] = final.groupby('seed')['r'].mean().values

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curves (Median & Q1–Q3)")
    ax.legend()
    fig.savefig(f"{output_prefix}_learning_curve.png")
    plt.close(fig)

    fig2, ax2 = plt.subplots()
    data = [boxplot_data['PPO'], boxplot_data['DQN']]
    ax2.boxplot(data, labels=['PPO', 'DQN'])
    ax2.set_ylabel("Reward")
    ax2.set_title("Final Performance Boxplot")
    fig2.savefig(f"{output_prefix}_boxplot.png")
    plt.close(fig2)

    stat, p = mannwhitneyu(boxplot_data['PPO'], boxplot_data['DQN'])
    print(f"Mann–Whitney U test: U={stat:.3f}, p-value={p:.5f}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline RL VizDoom: correr y analizar")
    parser.add_argument('--run', action='store_true', help="Ejecutar experimentos")
    parser.add_argument('--analyze', action='store_true', help="Analizar resultados")
    parser.add_argument('--algorithm', type=str, choices=['PPO', 'DQN'], help="Algoritmo para --run")
    parser.add_argument('--n_seeds', type=int, default=50, help="Número de runs independientes")
    parser.add_argument('--timesteps', type=int, default=100000, help="Timesteps por run")
    parser.add_argument('--scenario_cfg', type=str,
                        default="../ViZDoom/scenarios/defend_the_center.cfg",
                        help="Ruta al escenario VizDoom")
    parser.add_argument('--num_actions', type=int, default=3, help="Número de acciones discretas")
    parser.add_argument('--render', action='store_true', help="Renderizar entorno")
    parser.add_argument('--bin_size', type=int, default=5000, help="Tamaño de bin para análisis")
    parser.add_argument('--output_prefix', type=str, default="results", help="Prefijo para archivos de salida (.png)")

    args = parser.parse_args()
    if args.run:
        if not args.algorithm:
            parser.error("--run requiere --algorithm {PPO,DQN}")
        run_experiments(args.algorithm, args.n_seeds, args.timesteps,
                        args.scenario_cfg, args.num_actions, args.render)
    if args.analyze:
        analyze_and_plot(args.n_seeds, args.bin_size, args.output_prefix)


if __name__ == "__main__":
    main()