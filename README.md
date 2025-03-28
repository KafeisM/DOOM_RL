# 🧠 Reinforcement Learning en Doom con VizDoom

Este proyecto de Trabajo de Fin de Grado (TFG) explora el uso de algoritmos de _Reinforcement Learning (RL)_ aplicados al entorno de VizDoom — una herramienta que permite simular escenarios del videojuego Doom para investigación en inteligencia artificial.

El objetivo principal es analizar, comparar y entrenar agentes inteligentes que aprendan a actuar de forma autónoma en entornos complejos, usando distintas técnicas de RL proporcionadas por la librería [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

---

## 🎮 Herramientas principales

- **[VizDoom](https://github.com/mwydmuch/ViZDoom)** – Simulador de entornos tipo Doom para RL.
- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** – Implementaciones de algoritmos SOTA en RL.
- **Python** – Lenguaje principal del proyecto.
- **Gymnasium** – Interfaz de entornos de RL (adaptación personalizada para VizDoom).
- **WandB / TensorBoard** _(opcional)_ – Para seguimiento de métricas y visualización de entrenamientos.

---

## 📊 Resultados

Los agentes se entrenan en diferentes entornos (como `Basic`, `Death Corridor`, `Defend the center`) y se comparan en términos de:

- Tasa de supervivencia
- Recompensa media
- Aprendizaje en función del tiempo
- Estabilidad del entrenamiento
