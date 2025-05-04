from gymnasium import RewardWrapper

class ShapedRewardWrapper(RewardWrapper):
    """
    Reward shaping inspirado en el notebook de Github:
      - movement_reward: recompensa base por +Δx (distance-to-vest)
      - +10 × (damage_taken_delta): penaliza cada punto de salud perdido
      - +200 × (hitcount_delta): recompensa muy fuerte por cada hit/killcount
      - +5 × (ammo_delta): penaliza el uso de munición sin hits
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_damage_taken = 0
        self.prev_hitcount = 0
        self.prev_ammo = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        state = self.unwrapped.game.get_state()
        if state is not None and state.game_variables is not None:
            # [HEALTH, DAMAGE_TAKEN, HITCOUNT, SELECTED_WEAPON_AMMO]
            _, dmg_taken, hitcount, ammo = state.game_variables
            self.prev_damage_taken = dmg_taken
            self.prev_hitcount      = hitcount
            self.prev_ammo          = ammo
        return obs, info

    def reward(self, reward):
        """
        reward: movement_reward (+Δx or −Δx from base Doom)
        """
        state = self.unwrapped.game.get_state()
        if state is None or state.game_variables is None:
            return reward

        _, dmg_taken, hitcount, ammo = state.game_variables

        # deltas
        damage_taken_delta = -dmg_taken + self.prev_damage_taken
        hitcount_delta     = hitcount - self.prev_hitcount
        ammo_delta         = ammo - self.prev_ammo

        # shaped reward
        shaped = reward
        shaped += 10.0 * damage_taken_delta    # −10 por cada punto de daño
        shaped += 200.0 * hitcount_delta       # +200 por cada hit/killcount
        shaped += 5.0 * ammo_delta             # −5 por cada bala disparada sin hit

        # actualizar previos
        self.prev_damage_taken = dmg_taken
        self.prev_hitcount      = hitcount
        self.prev_ammo          = ammo

        return shaped
