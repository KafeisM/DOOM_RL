import time
from stable_baselines3 import PPO
from basic import VizDoomEnv


def main():
    model = PPO.load("train/train_basic/best_model_100000.zip")

    env =  VizDoomEnv(render=True)
    print(env.observation_space)

    # Evaluate mean reward for n games, use with render false
    # mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

    obs = env.reset()
    for episode in range(20):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)

            obs, reward, done, _, info = env.step(action)
            time.sleep(0.05)
            total_reward += reward
        print('Total Reward for episode {} is {}'.format(episode, total_reward))
        time.sleep(1)

if __name__ == '__main__':
    main()