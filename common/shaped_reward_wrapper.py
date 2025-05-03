from gymnasium import RewardWrapper

class ShapedRewardWrapper(RewardWrapper):
    """
    Reward shaping for VizDoom Deadly Corridor:
      - +5.0 per kill (Δ KILLCOUNT)
      - +0.1 per point of damage inflicted (Δ DAMAGECOUNT)
      - -0.1 per point of damage received (DAMAGE_TAKEN)
      - -0.1 per missed shot (ammo used without damage or kills)
      - +0.05 per forward movement (reward base > 0)
      - -0.1 per timestep of stasis (>5 timesteps without damage or kills)
    """
    def __init__(self, env):
        super().__init__(env)
        self.prev_kills = 0
        self.prev_dmg   = 0
        self.prev_ammo  = 0
        self.step_counter = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_counter = 0
        state = self.unwrapped.game.get_state()
        if state is not None and state.game_variables is not None:
            # [HEALTH, DAMAGE_TAKEN, DAMAGECOUNT, KILLCOUNT, SELECTED_WEAPON_AMMO]
            _, _, dmg, kills, ammo = state.game_variables
            self.prev_kills = kills
            self.prev_dmg   = dmg
            self.prev_ammo  = ammo
        return obs, info

    def reward(self, reward):
        state = self.unwrapped.game.get_state()
        if state is None or state.game_variables is None:
            return reward

        _, dmg_taken, dmg_count, kills, ammo = state.game_variables
        shaped = reward

        # 1) Bonus per kill
        delta_kills = kills - self.prev_kills
        if delta_kills > 0:
            shaped += 5.0 * delta_kills

        # 2) Bonus per damage inflicted
        delta_dmg = dmg_count - self.prev_dmg
        if delta_dmg > 0:
            shaped += 0.1 * delta_dmg

        # 3) Penalty per damage received
        shaped -= 0.1 * dmg_taken

        # 4) Penalty per missed shot
        delta_ammo = self.prev_ammo - ammo
        if delta_ammo > 0 and delta_dmg == 0 and delta_kills == 0:
            shaped -= 0.1 * delta_ammo

        # 5) Small bonus for forward movement
        if reward > 0:
            shaped += 0.05

        # 6) Penalty for stasis (>5 timesteps without damage or kills)
        self.step_counter += 1
        if self.step_counter > 5 and delta_dmg == 0 and delta_kills == 0:
            shaped -= 0.1

        # update previous values
        self.prev_kills = kills
        self.prev_dmg   = dmg_count
        self.prev_ammo  = ammo

        return shaped