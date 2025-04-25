# test_envs.py
from gymnasium.utils.env_checker import check_env
from common import envs

def main():
    env = envs.create_env(scenario="basic")
    check_env(env)  # Validación automática de Gymnasium
    obs, info = env.reset()
    print("Obs shape:", obs.shape)
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Reward: {reward}, Done: {terminated or truncated}")
        if terminated or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    main()
