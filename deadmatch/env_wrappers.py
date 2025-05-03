import numpy as np
import cv2
import gymnasium as gym
from gymnasium import ObservationWrapper, Wrapper
from gymnasium.spaces import Box
from collections import deque
from gymnasium import ActionWrapper
from gymnasium.spaces import Discrete, Dict as SpaceDict


class PreprocessFrame(ObservationWrapper):
    def __init__(self, env, width=108, height=60, grayscale=False):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        num_channels = 1 if grayscale else 3
        # Definimos el espacio de observación como imágenes normalizadas en [0,1]
        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=(height, width, num_channels),
            dtype=np.float32
        )

    def observation(self, obs):
        # Extraemos la pantalla del dict
        img = obs['screen']

        # Si viene en formato (C, H, W), lo pasamos a (H, W, C)
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        # Ahora img tiene forma (H, W, C)
        # Convertimos a gris si es necesario
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Redimensionamos a (width, height)
        img_resized = cv2.resize(
            img,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA
        )

        # Si es gris, añadir dimensión de canal
        if self.grayscale:
            img_resized = np.expand_dims(img_resized, axis=-1)

        # Normalizamos a [0,1]
        img_norm = img_resized.astype(np.float32) / 255.0

        return img_norm

class FrameStack(Wrapper):
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        ch = env.observation_space.shape[2]
        h = env.observation_space.shape[0]
        w = env.observation_space.shape[1]
        new_channels = ch * n_frames
        self.observation_space = Box(low=0.0, high=1.0,
                                     shape=(h, w, new_channels),
                                     dtype=np.float32)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.frames.clear()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        return np.concatenate(list(self.frames), axis=2)

class RewardShapingWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_health = None
        self.prev_ammo = None
        self.prev_kills = None

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if 'gamevariables' in obs:
            gv = obs['gamevariables']
            self.prev_health = int(gv[0])
            self.prev_ammo = int(gv[3])
            self.prev_kills = int(gv[4])
        else:
            self.prev_health = 100
            self.prev_ammo = 0
            self.prev_kills = 0
        return obs, info

    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        shaped = 0.0
        if 'gamevariables' in obs:
            gv = obs['gamevariables']
            health, ammo, kills = int(gv[0]), int(gv[3]), int(gv[4])
        else:
            health, ammo, kills = self.prev_health, self.prev_ammo, self.prev_kills
        # Frag reward
        if kills > self.prev_kills:
            shaped += 1.0 * (kills - self.prev_kills)
        # Ammo pickup
        if ammo > self.prev_ammo:
            shaped += 0.02 * (ammo - self.prev_ammo)
        # Ammo spent
        if ammo < self.prev_ammo:
            shaped -= 0.01 * (self.prev_ammo - ammo)
        # Health pickup
        if health > self.prev_health:
            shaped += 0.02 * (health - self.prev_health)
        # Damage taken
        if health < self.prev_health:
            shaped -= 0.01 * (self.prev_health - health)
        # Movement incentive/penalty
        if action in [0, 9]:
            shaped -= 0.0025
        else:
            shaped += 0.0005
        self.prev_health = health
        self.prev_ammo = ammo
        self.prev_kills = kills
        return obs, base_reward + shaped, terminated, truncated, info

class DiscreteBinaryActionWrapper(ActionWrapper):
    """
    Toma un env cuya action_space es Dict({'binary':Discrete(n), 'continuous':Box(3) })
    y expone solo el Discrete(n). Cuando el agente elige act (un int),
    lo envuelve de nuevo en {'binary': act, 'continuous': zeros(3)}.
    """
    def __init__(self, env):
        super().__init__(env)
        asp = env.action_space
        # Comprobar formato esperado
        if (isinstance(asp, SpaceDict)
            and 'binary' in asp.spaces
            and 'continuous' in asp.spaces
            and isinstance(asp.spaces['binary'], Discrete)
            and isinstance(asp.spaces['continuous'], Box)
            and asp.spaces['continuous'].shape == (3,)):
            # Exponer solo el Discrete de 'binary'
            self.action_space = asp.spaces['binary']
            # Guardamos la forma del continuous para rellenar ceros
            self._cont_shape = asp.spaces['continuous'].shape
        else:
            raise ValueError(
                f"Se esperaba Dict('binary':Discrete, 'continuous':Box(3)), "
                f"pero se obtuvo {asp}"
            )

    def action(self, act: int):
        # construye el dict con la parte continua a ceros
        return {
            'binary': int(act),
            'continuous': np.zeros(self._cont_shape, dtype=self.env.action_space.dtype)
        }

