from stable_baselines3 import DQN, PPO, A2C
import os
MODEL_PATH = "./PPO/train - models/train_deadly_corridor/best_model/best_model.zip"
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"No se encontró el modelo en: {MODEL_PATH}")
        return

    model = PPO.load(MODEL_PATH)

    print("\n=== Arquitectura de la política (policy) ===")
    print(model.policy)

    print("\n=== Extractor de características (features_extractor) ===")
    print(model.policy.features_extractor)

if __name__ == "__main__":
    main()