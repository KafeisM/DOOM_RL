import vizdoom as vzd
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import cv2
import numpy as np

# Create Vizdoom OpenAI Gym Environment
class VizDoomGymReward(gym.Env): 
    # Function that is called when we start the env
    def __init__(self, render=False, config='./Scenarios/deadly_corridor/deadly_corridor - t1.cfg'): 
        # Inherit from Env
        super().__init__()
        # Setup the game 
        self.game = vzd.DoomGame()
        self.game.load_config(config)
        
        # Render frame logic
        if render:
            self.game.set_window_visible(True)
        else:
            self.game.set_window_visible(False)
        
        # Start the game 
        self.game.init()
        
        # Create the action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8) 
        self.action_space = Discrete(7)
        
        # Game variables: HEALTH DAMAGE_TAKEN HITCOUNT SELECTED_WEAPON_AMMO
        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 52 ## CHANGED
        
        
    # This is how we take a step in the environment
    def step(self, action):
        # Specify action and take step
        actions = np.identity(7)
        movement_reward = self.game.make_action(actions[action], 4)

        # Get all the other stuff we need to return
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)

            # Reward shaping
            game_variables = self.game.get_state().game_variables
            health, damage_taken, hitcount, ammo = game_variables

            # Calculate reward deltas
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.ammo
            self.ammo = ammo

            # Recompensa modificada para fomentar comportamientos más complejos
            reward = (
                    movement_reward * 0.4 +
                    damage_taken_delta * 15 +
                    hitcount_delta * 250 +
                    ammo_delta * (-5)  # Penalizar uso de munición sin resultados
            )

            # Bonificación por eficiencia: premiar aciertos con disparos
            if hitcount_delta > 0:
                reward += 25  # Bonificación adicional por acertar

            # Penalización por muerte
            if health <= 0:
                reward -= 100

            info = ammo
        else:
            state = np.zeros(self.observation_space.shape)
            reward = 0  # Asignar valor por defecto a reward cuando no hay estado
            info = 0

        info = {"info": info}
        done = self.game.is_episode_finished()
        truncated = False

        return state, reward, done, truncated, info
    
    # Define how to render the game or environment 
    def render(self):
        pass
    
    # What happens when we start a new game 
    def reset(self, seed=None, options=None):
        # Establecer la semilla si se proporciona
        if seed is not None:
            np.random.seed(seed)

        self.game.new_episode()

        # Restablecer las variables de juego
        if self.game.get_state():
            state = self.game.get_state().screen_buffer

            # También puedes restablecer tus variables de seguimiento aquí
            game_variables = self.game.get_state().game_variables
            _, self.damage_taken, self.hitcount, self.ammo = game_variables

            return self.grayscale(state), {}  # Gymnasium espera que reset devuelva (obs, info)
        else:
            # Manejar el caso cuando no hay estado disponible
            return np.zeros(self.observation_space.shape), {}
    
    # Grayscale the game frame and resize it 
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state
    
    # Call to close down the game
    def close(self): 
        self.game.close()