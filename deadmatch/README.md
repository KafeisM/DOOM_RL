# Doom Deathmatch RL Agent

Proyecto de Deep RL para ViZDoom Deathmatch usando Gymnasium y PyTorch.

## Estructura

- `env_wrappers.py` – Wrappers de Gym (preprocesamiento, stacking, reward shaping).
- `model.py` – Definición de la red DQN.
- `utils.py` – ReplayMemory y epsilon scheduling.
- `train.py` – Script de entrenamiento.
- `evaluate.py` – Script de evaluación.
- `requirements.txt` – Dependencias.
- `README.md` – Documentación del proyecto.

## Uso

1. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
2. Entrenar el agente:
   ```
   python train.py
   ```
3. Evaluar el agente entrenado:
   ```
   python evaluate.py policy_net.pth
   ```
