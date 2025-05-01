# shaped_reward_wrapper.py (versión optimizada con DAMAGECOUNT)
from gymnasium import RewardWrapper

class ShapedRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_health = None
        self.prev_damage_count = None
        self.prev_ammo = None
        self.step_counter = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.step_counter = 0

        try:
            if self.unwrapped.game.get_state() is not None:
                game_vars = self.unwrapped.game.get_state().game_variables
                self.prev_health = game_vars[0]     # HEALTH
                self.prev_damage_count = game_vars[2]  # DAMAGECOUNT
                self.prev_ammo = game_vars[3]       # AMMO
        except Exception:
            self.prev_health = None
            self.prev_damage_count = None
            self.prev_ammo = None

        return observation, info

    def reward(self, reward):
        if self.unwrapped.game.get_state() is None:
            return reward

        try:
            game_vars = self.unwrapped.game.get_state().game_variables
            if game_vars is None or len(game_vars) < 4:
                return reward

            # Variables del juego
            health = game_vars[0]             # HEALTH
            damage_taken = game_vars[1]       # DAMAGE_TAKEN
            damage_count = game_vars[2]       # DAMAGECOUNT
            ammo = game_vars[3]               # AMMO

            shaped = reward

            # 1. Penalización por perder salud
            if self.prev_health is not None:
                delta_health = health - self.prev_health
                if delta_health < 0:
                    shaped -= 0.3 * abs(delta_health)

            # 2. Recompensa por daño infligido real
            if self.prev_damage_count is not None:
                delta_damage = damage_count - self.prev_damage_count
                shaped += 0.1 * delta_damage

            # 3. Penalización por disparar sin impactar
            if self.prev_ammo is not None:
                delta_ammo = self.prev_ammo - ammo
                if delta_ammo > 0 and delta_damage == 0:
                    shaped -= 0.05 * delta_ammo

            # 4. Bonus por supervivencia
            shaped += 0.01

            # 5. Penalización por estancamiento prolongado
            self.step_counter += 1
            if self.step_counter > 100 and reward <= 0:
                shaped -= 0.05

            self.prev_health = health
            self.prev_damage_count = damage_count
            self.prev_ammo = ammo

            return shaped

        except Exception:
            return reward
