import os
import pandas as pd
import matplotlib.pyplot as plt

# Directorio base donde están las carpetas seed_0, seed_1, etc.
base_dir = "logs/DQN/"

# Recorrer todas las carpetas que empiezan con "seed_"
for seed_dir in sorted([d for d in os.listdir(base_dir) if d.startswith("seed_")]):
    # Construir la ruta completa al archivo monitor.csv
    csv_path = os.path.join(base_dir, seed_dir, "monitor.csv")

    # Verificar si existe el archivo
    if os.path.exists(csv_path):
        try:
            # Cargar el CSV saltando la primera línea (comentario)
            df = pd.read_csv(csv_path, skiprows=1)

            # Obtener la última recompensa
            ultimo_reward = df["r"].iloc[-1]

            # Imprimir el resultado
            print(f"Recompensa final de {seed_dir}: {ultimo_reward}")
        except Exception as e:
            print(f"Error al procesar {seed_dir}: {e}")
    else:
        print(f"No se encontró el archivo monitor.csv en {seed_dir}")