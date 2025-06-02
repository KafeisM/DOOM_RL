#!/usr/bin/env python3
"""
analyze_ppo.py

Script independiente para analizar los resultados de PPO (n_SEEDS runs) sin tocar el código original.
Genera:
  - Curva de aprendizaje (mediana + banda Q1–Q3) de PPO sobre timesteps acumulados.
  - Boxplot del rendimiento final (promedio del último bin) de PPO (por seed).

Uso:
  python analyze_ppo.py --n_seeds 50 --bin_size 5000 --output_prefix estudioPPO
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_data_cumulative(algorithm: str, n_seeds: int) -> pd.DataFrame:
    """
    Para cada `seed` (0..n_seeds-1), abre:
        logs/{algorithm}/seed_{seed}/monitor.csv

    Cada monitor.csv:
      - Tiene una primera línea de comentario (que ignoramos con skiprows=1)
      - Luego la cabecera: r, l, t  (reward, length, tiempo)

    Construye un DataFrame concatenado con columnas:
      ['seed', 'cum_steps', 'r']

    Donde:
      - 'r' es la recompensa por episodio
      - 'cum_steps' es la suma acumulada de 'l' (episodio a episodio) para esa semilla
      - 'seed' es el índice de semilla (0..n_seeds-1)
    """
    all_runs = []

    for seed in range(n_seeds):
        path = os.path.join("logs", algorithm, f"seed_{seed}", "monitor.csv")
        if not os.path.exists(path):
            print(f"[WARNING] No se encontró monitor.csv en: {path}  → se ignora esta seed")
            continue

        # Leemos, saltando la primera línea de comentario '# {...}'
        df = pd.read_csv(path, skiprows=1)

        # Asegurarnos de que exista la columna 'l' y 'r'
        if 'l' not in df.columns or 'r' not in df.columns:
            raise RuntimeError(f"El archivo {path} no tiene columnas 'l' y 'r'. "
                               "Revisa que sea un Monitor típico de Stable-Baselines.")

        # Calculamos timesteps acumulados para esta seed:
        df = df[['l', 'r']].copy()
        df['cum_steps'] = df['l'].cumsum()

        # Añadimos el índice de semilla
        df['seed'] = seed

        # Quedarnos solo con (seed, cum_steps, r)
        all_runs.append(df[['seed', 'cum_steps', 'r']])

    if len(all_runs) == 0:
        raise FileNotFoundError(f"No se encontraron datos para el algoritmo {algorithm} "
                                f"en ninguna de las {n_seeds} seeds.")

    # Concatenamos todas las seeds en un solo DataFrame
    return pd.concat(all_runs, ignore_index=True)


def analyze_ppo_only(n_seeds: int, bin_size: int, output_prefix: str):
    """
    1) Carga todos los CSV de PPO y calcula timesteps acumulados por seed.
    2) Construye bins de tamaño `bin_size` sobre 'cum_steps' (ej.: 0–4999, 5000–9999, …).
    3) Para cada bin, calcula mediana de 'r' y cuartiles (Q1, Q3) sobre todas las seeds.
       → Gráfica curva de aprendizaje (mediana + Q1–Q3).
    4) Para cada seed, extrae los episodios cuyo cum_steps estén en el último bin:
       [max_cum_steps_seed – bin_size, max_cum_steps_seed], y promedia su recompensa 'r'.
       Con esos 𝑛 valores (uno por seed) arma un boxplot.
    5) Imprime en consola estadísticas básicas (media, mediana y desviación) del conjunto final.
    """
    algorithm = "PPO"
    print(f">>> Cargando datos de '{algorithm}' para {n_seeds} seeds…")
    df = load_data_cumulative(algorithm, n_seeds)
    # df tiene columnas: ['seed', 'cum_steps', 'r']

    # 1) Definir a qué bin corresponde cada episodio (bin de timesteps)
    df['step_bin'] = (df['cum_steps'] // bin_size) * bin_size

    # 2) Agrupar por bin y sacar estadísticos (mediana, Q1, Q3) de 'r' en cada bin
    stats = df.groupby('step_bin')['r'].agg(
        median='median',
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75)
    ).reset_index()

    # ——— Gráfico 1: Curva de aprendizaje — mediana + banda Q1–Q3 ———
    print("Generando curva de aprendizaje (mediana + Q1–Q3)…")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(stats['step_bin'], stats['median'], color='blue', label=f"{algorithm} mediana")
    ax.fill_between(stats['step_bin'], stats['q1'], stats['q3'],
                    color='blue', alpha=0.3, label=f"{algorithm} Q1–Q3")
    ax.set_xlabel("Timesteps acumulados")
    ax.set_ylabel("Recompensa por episodio")
    ax.set_title(f"Curva de aprendizaje de {algorithm} (Mediana & Q1–Q3)")
    ax.legend()
    fig.tight_layout()
    learning_curve_path = f"{output_prefix}_{algorithm}_learning_curve.png"
    fig.savefig(learning_curve_path)
    plt.close(fig)
    print(f"  • Curva guardada en: {learning_curve_path}")

    # ——— Gráfico 2: Boxplot del rendimiento final por seed ———
    print("Generando boxplot del rendimiento final (último bin) por seed…")
    # Para cada seed, hallamos su máximo de cum_steps
    boxplot_vals = []
    for seed in sorted(df['seed'].unique()):
        df_seed = df[df['seed'] == seed]
        max_steps_seed = df_seed['cum_steps'].max()
        # Filtrar los episodios en [max_steps_seed - bin_size, max_steps_seed]
        threshold = max_steps_seed - bin_size
        last_bin_df = df_seed[df_seed['cum_steps'] > threshold]
        if last_bin_df.shape[0] == 0:
            # Si no hay ningún episodio completo en ese rango, usamos el último valor disponible
            last_reward = df_seed.iloc[-1]['r']
        else:
            # Promediar recompensa en esos episodios
            last_reward = last_bin_df['r'].mean()
        boxplot_vals.append(last_reward)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.boxplot(boxplot_vals, labels=[algorithm])
    ax2.set_ylabel("Recompensa promedio en el último bin")
    ax2.set_title(f"Distribución del rendimiento final de {algorithm} (n={len(boxplot_vals)})")
    fig2.tight_layout()
    boxplot_path = f"{output_prefix}_{algorithm}_boxplot.png"
    fig2.savefig(boxplot_path)
    plt.close(fig2)
    print(f"  • Boxplot guardado en: {boxplot_path}")

    # ——— Estadísticas finales impresas en consola ———
    serie_final = pd.Series(boxplot_vals)
    mean_val = serie_final.mean()
    median_val = serie_final.median()
    std_val = serie_final.std(ddof=0)
    print("\n— Estadísticas finales (en el último bin) —")
    print(f"  • Media   : {mean_val:.2f}")
    print(f"  • Mediana : {median_val:.2f}")
    print(f"  • Desv. Std: {std_val:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analizar resultados de PPO (n seeds) calculando timesteps acumulados."
    )
    parser.add_argument(
        '--n_seeds', type=int, default=50,
        help="Número de runs (seeds) independientes (por defecto: 50)."
    )
    parser.add_argument(
        '--bin_size', type=int, default=5000,
        help="Tamaño del bin sobre timesteps acumulados (por defecto: 5000)."
    )
    parser.add_argument(
        '--output_prefix', type=str, default="resultadoPPO",
        help="Prefijo para los archivos .png que se generarán (por defecto: 'resultadoPPO')."
    )

    args = parser.parse_args()

    try:
        analyze_ppo_only(args.n_seeds, args.bin_size, args.output_prefix)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        exit(1)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        exit(1)


if __name__ == "__main__":
    main()
