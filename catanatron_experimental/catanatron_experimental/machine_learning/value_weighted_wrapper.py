"""
Value-Weighted Imitation Wrapper.

Instead of binary reward for matching the expert's top action (v10),
this wrapper scores ALL valid actions using the ValueFunction heuristic
and gives a proportional reward based on action quality.

Key advantage over ExpertGuidedWrapper (v10):
  - v10: binary signal (+bonus if action == expert_argmax, else +0)
         Agent gets ZERO signal unless it randomly picks the expert's action.
  - v13: continuous signal (+bonus * normalized_score)
         Agent ALWAYS gets signal. Picking 2nd-best still yields high bonus.
         This makes learning much more sample-efficient.

Normalization:
  - Best valid action  -> normalized_score = 1.0 -> bonus = bonus_scale
  - Worst valid action -> normalized_score = 0.0 -> bonus = 0.0
  - In-between actions get proportional credit
"""

import gymnasium as gym
from catanatron_gym.envs.catanatron_env import to_action_space
from catanatron_experimental.machine_learning.players.value import base_fn


class ValueWeightedWrapper(gym.Wrapper):
    """
    Gym wrapper that gives reward proportional to action quality
    as judged by the ValueFunction heuristic.

    Args:
        env: The base Catan environment (already wrapped with ActionMasker)
        player_color: Color of the agent in the game (for value_fn evaluation)
        bonus_scale: Max bonus for picking the best action (default: 0.2)
        decay_start_step: Step at which to start decaying bonus_scale
        decay_rate: Per-step multiplicative decay after decay_start_step
        min_scale: Floor for bonus_scale after decay
        bonus_phases: Optional list of (step, bonus) tuples for phase-based decay.
                      If provided, overrides exponential decay. Linearly interpolates
                      between phases. Steps are per-env steps.
    """

    def __init__(
        self,
        env,
        player_color,
        bonus_scale=0.2,
        decay_start_step=8_000_000,
        decay_rate=0.99999,
        min_scale=0.05,
        bonus_phases=None,
    ):
        super().__init__(env)
        self.player_color = player_color
        self.bonus_scale = bonus_scale
        self.initial_scale = bonus_scale
        self.decay_start_step = decay_start_step
        self.decay_rate = decay_rate
        self.min_scale = min_scale
        self.bonus_phases = bonus_phases
        self.value_fn = base_fn()

        # Statistics
        self.steps = 0
        self.total_decisions = 0
        self.score_sum = 0.0
        self.best_action_count = 0

    def step(self, action):
        game = self.env.unwrapped.game
        playable_actions = game.state.playable_actions

        bonus = 0.0
        normalized_score = None

        if len(playable_actions) > 1:
            self.total_decisions += 1

            try:
                # Score all valid actions using the value function
                action_scores = {}
                for pa in playable_actions:
                    game_copy = game.copy()
                    game_copy.execute(pa)
                    action_idx = to_action_space(pa)
                    action_scores[action_idx] = self.value_fn(
                        game_copy, self.player_color
                    )

                min_val = min(action_scores.values())
                max_val = max(action_scores.values())

                # Look up agent's chosen action value
                agent_value = action_scores.get(action, min_val)

                if max_val > min_val:
                    normalized_score = (agent_value - min_val) / (max_val - min_val)
                else:
                    normalized_score = 1.0  # All actions are equivalent

                bonus = self.bonus_scale * normalized_score
                self.score_sum += normalized_score

                if normalized_score > 0.99:
                    self.best_action_count += 1

            except Exception:
                # If scoring fails, give no bonus (don't crash training)
                bonus = 0.0

        # Execute the action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add value-weighted bonus to reward
        reward += bonus

        # Decay bonus_scale
        self.steps += 1
        if self.bonus_phases is not None:
            # Phase-based linear interpolation
            for i in range(len(self.bonus_phases) - 1):
                s0, b0 = self.bonus_phases[i]
                s1, b1 = self.bonus_phases[i + 1]
                if self.steps <= s1:
                    frac = (self.steps - s0) / max(1, s1 - s0)
                    frac = max(0.0, min(1.0, frac))
                    self.bonus_scale = b0 + (b1 - b0) * frac
                    break
            else:
                # Past all phases
                self.bonus_scale = self.bonus_phases[-1][1]
        elif self.steps > self.decay_start_step:
            self.bonus_scale = max(
                self.min_scale, self.bonus_scale * self.decay_rate
            )

        # Stats for logging / WandB
        info["value_bonus"] = bonus
        info["value_bonus_scale"] = self.bonus_scale
        if normalized_score is not None:
            info["value_normalized_score"] = normalized_score
        if self.total_decisions > 0:
            info["value_avg_score"] = self.score_sum / self.total_decisions
            info["value_best_action_rate"] = (
                self.best_action_count / self.total_decisions
            )

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
