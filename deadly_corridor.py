# Imports de bibliotecas estándar
import os
import time
import random
# Imports de bibliotecas científicas
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Imports de VizDoom
import vizdoom as vzd

# Imports de Gymnasium
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

# Imports de Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

def main():
    # Setup game
    game = vzd.DoomGame()
    game.load_config("Scenarios/deadly_corridor/deadly_corridor - t5.cfg")
    
    # Inicializar el juego antes de obtener cualquier estado
    game.init()

    # This is the set of actions we can take in the environment
    actions = np.identity(7, dtype=np.uint8)
    
    # Loop thorugh episodes
    episodes = 10
    for episode in range(episodes):
        # Create a new episode or game
        game.new_episode()
        
        # Ahora podemos obtener el estado de manera segura
        state = game.get_state()
        if state is not None:
            print(state.game_variables)
            
        # Checking hte game isn't finish
        while not game.is_episode_finished():
            # Get the game state
            state = game.get_state()
            # Get the game image
            img = state.screen_buffer
            # Get the game variables - ammo
            info = state.game_variables
            # Take an action
            reward = game.make_action(random.choice(actions), 7)
            # Print reward
            print('Reward:', reward)
            time.sleep(0.02)
        print('Result:', game.get_total_reward())
        time.sleep(2)

    game.close()

if __name__ == '__main__':
    main()