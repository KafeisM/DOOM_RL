# DOOM\_RL: Aprendizaje por Refuerzo en Entornos VizDoom

Este repositorio contiene la implementación, experimentación y análisis de algoritmos de **Aprendizaje por Refuerzo Profundo (DRL)** aplicados a escenarios del videojuego DOOM mediante la plataforma **VizDoom**. Además de proveer el código, se incluyen los scripts de experimentación masiva y las figuras generadas para el Trabajo de Fin de Grado (TFG) “Aprenentatge per Reforç aplicat a entorns 3D”.

## Visión General

El objetivo del proyecto es **comparar empíricamente** dos familias de algoritmos DRL —*Deep Q‑Network (DQN)* y *Proximal Policy Optimization (PPO)*— en tres escenarios 3D con dificultad creciente. El estudio analiza la **velocidad de convergencia, estabilidad y rendimiento asintótico** de cada método y extrae conclusiones sobre su idoneidad según la naturaleza del entorno.

## Estructura del Proyecto

```text
DOOM_RL/
├── common/                  # Módulos reutilizables (entornos, callbacks, wrappers)
├── DQN/                     # Entrenamientos DQN
├── PPO/                     # Entrenamientos PPO
├── runs/                    # Pipelines de experimentación y análisis
├── Scenarios/               # Configuración .cfg/.wad de VizDoom
├── train-models/            # Checkpoints y mejores modelos guardados
├── logs/                    # Métricas de TensorBoard (50 seeds × config)
└── docs/                    # Figuras y tablas usadas en la memoria
```

## Escenarios Implementados

| Escenario             | Objetivo                                                                  | Acciones                                | Recompensa Clave                                    |
| --------------------- | ------------------------------------------------------------------------- | --------------------------------------- | --------------------------------------------------- |
| **Basic**             | Disparar a un único enemigo estático en una sala rectangular.             | `MOVE_LEFT`, `MOVE_RIGHT`, `ATTACK`     | +106 por kill, −5 por disparo, +1 por tic vivo      |
| **Defend the Center** | Mantenerse con vida en un área circular eliminando oleadas de 5 enemigos. | `TURN_LEFT`, `TURN_RIGHT`, `ATTACK`     | +1 por kill, −1 al morir                            |
| **Deadly Corridor**   | Alcanzar una armadura al final de un pasillo bajo fuego enemigo.          | 7 combinaciones de movimiento + disparo | +Δdistancia al objetivo, −Δdistancia, −100 al morir |

> Los archivos `.cfg` de cada escenario se encuentran en **Scenarios/** y pueden modificarse para ajustar dificultad, tiempo límite o recompensas.

## Metodología Experimental

El **pipeline** en `runs/experiment_pipeline_DC.py` automatiza entrenamiento, checkpoints y post‑procesado; permite reanudar experimentos y exporta los resultados a **TensorBoard** y CSV para análisis posterior.

## Instalación y Requisitos

```bash
git clone https://github.com/KafeisM/DOOM_RL.git
pip install -r requirements.txt  # Incluye gymnasium vizdoom stable-baselines3 pytorch...
```
Sigue las instrucciones oficiales de [VizDoom](https://github.com/mwydmuch/ViZDoom) para compilar el motor y asegurarte de que las librerías nativas estén en tu `LD_LIBRARY_PATH`.

## Documentación

Puedes consultar la memoria del TFG completo del proyecto en el siguiente documento:

[Abrir documento PDF](./TFG_TL_DOOM_FINAL.pdf)

## Referencias y Recursos

- [VizDoom](https://github.com/mwydmuch/ViZDoom)
- [Stable‑Baselines 3](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gymnasium](https://gymnasium.farama.org/)
- [TensorBoard](https://www.tensorflow.org/tensorboard?hl=es-419)

