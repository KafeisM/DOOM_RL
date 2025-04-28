from gymnasium import RewardWrapper
import numpy as np


class ShapedRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_health = None
        self.prev_hitcount = None
        self.prev_ammo = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # Manejo seguro de game_variables
        try:
            if self.unwrapped.game.get_state() is not None:
                game_vars = self.unwrapped.game.get_state().game_variables
                self.prev_health = game_vars[0]
                self.prev_hitcount = game_vars[2]  # Cambiar a los índices correctos
                self.prev_ammo = game_vars[3]
        except (AttributeError, IndexError) as e:
            print(f"⚠️ Error al acceder a game_variables: {e}")
            self.prev_health = None
            self.prev_hitcount = None
            self.prev_ammo = None

        return observation, info

    def reward(self, reward):
        # Verificar si el estado existe
        if self.unwrapped.game.get_state() is None:
            return reward

        try:
            game_vars = self.unwrapped.game.get_state().game_variables

            # Verificar que game_vars tenga suficientes elementos
            if game_vars is None or len(game_vars) < 4:
                return reward

            # Según la documentación/config, game_vars tiene este orden:
            # [HEALTH, DAMAGE_TAKEN, HITCOUNT, SELECTED_WEAPON_AMMO]
            health = game_vars[0]
            damage_taken = game_vars[1]
            hitcount = game_vars[2]
            ammo = game_vars[3]

            shaped = reward

            # Penalizar pérdida de vida
            if self.prev_health is not None:
                delta_health = health - self.prev_health
                shaped += 0.05 * delta_health

            # Dar reward extra por impacto
            if self.prev_hitcount is not None:
                delta_hits = hitcount - self.prev_hitcount
                shaped += 0.5 * delta_hits  # cada hit cuenta bastante

            # Penalizar daño recibido
            shaped -= 0.1 * damage_taken

            # Bonus pequeño si tiene munición
            shaped += 0.01 * ammo

            # Actualizar para siguiente paso
            self.prev_health = health
            self.prev_hitcount = hitcount
            self.prev_ammo = ammo

            return shaped

        except Exception as e:
            print(f"⚠️ Error en reward shaping: {e}")
            return reward