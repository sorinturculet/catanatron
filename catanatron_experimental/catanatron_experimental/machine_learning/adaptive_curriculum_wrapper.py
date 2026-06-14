"""
Adaptive Expert Curriculum Wrapper (v17).

Replaces fixed step-based decay of expert guidance with performance-gated
adaptive decay. The expert bonus decays faster when the agent demonstrates
higher competency (measured by rolling normalized_score), and slower when
the agent is struggling.

Key differences from ValueWeightedWrapper (v13/v15):
  - Warmup period followed by adaptive decay (not fixed step threshold)
  - Decay rate scales with agent competency (not constant)
  - min_scale=0.0 allows full graduation from expert guidance
  - Rich phase-tracking and logging for WandB visualization

Decay formula:
  competency = rolling_mean(recent_normalized_scores)
  effective_decay = base_decay_rate ^ (1.0 + competency_multiplier * competency)
  bonus_scale = max(min_scale, bonus_scale * effective_decay)

Phases:
  1. Warmup: Full expert guidance (bonus_scale constant)
  2. Adaptive: Decay rate proportional to agent competency
  3. Graduated: Expert fully removed, pure RL
"""

from collections import deque

import gymnasium as gym
from catanatron_gym.envs.catanatron_env import to_action_space
from catanatron_experimental.machine_learning.players.value import base_fn


class AdaptiveCurriculumWrapper(gym.Wrapper):
    """
    Gym wrapper with adaptive expert guidance decay.

    Scores all valid actions using VFP's value function (same as ValueWeightedWrapper),
    but decays the bonus based on the agent's rolling competency score rather than
    a fixed step count.

    Args:
        env: The base Catan environment (already wrapped with ActionMasker)
        player_color: Color of the agent in the game
        bonus_scale: Initial max bonus for picking the best action (default: 0.2)
        warmup_steps: Per-env steps of full guidance before adaptive decay begins
        base_decay_rate: Per-step multiplicative decay when competency=0
        competency_multiplier: How much competency accelerates decay
        performance_window: Size of rolling buffer for competency estimation
        min_scale: Floor for bonus_scale (0.0 = full graduation possible)
        min_samples: Minimum buffer entries before adaptive decay activates
        graduation_threshold: bonus_scale below this = graduated phase
    """

    def __init__(
        self,
        env,
        player_color,
        bonus_scale=0.2,
        warmup_steps=125_000,
        base_decay_rate=0.99999,
        competency_multiplier=3.0,
        performance_window=50_000,
        min_scale=0.0,
        min_samples=10_000,
        graduation_threshold=0.001,
    ):
        super().__init__(env)
        self.player_color = player_color
        self.bonus_scale = bonus_scale
        self.initial_scale = bonus_scale
        self.warmup_steps = warmup_steps
        self.base_decay_rate = base_decay_rate
        self.competency_multiplier = competency_multiplier
        self.min_scale = min_scale
        self.min_samples = min_samples
        self.graduation_threshold = graduation_threshold
        self.value_fn = base_fn()

        # Rolling buffer for competency estimation
        self.score_buffer = deque(maxlen=performance_window)

        # Statistics
        self.steps = 0
        self.total_decisions = 0
        self.score_sum = 0.0
        self.best_action_count = 0
        self.last_effective_decay = 1.0

    def step(self, action):
        game = self.env.unwrapped.game
        playable_actions = game.state.playable_actions

        bonus = 0.0
        normalized_score = None

        # Score all valid actions using VFP value function
        if len(playable_actions) > 1:
            self.total_decisions += 1

            try:
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

                agent_value = action_scores.get(action, min_val)

                if max_val > min_val:
                    normalized_score = (agent_value - min_val) / (max_val - min_val)
                else:
                    normalized_score = 1.0  # All actions equivalent

                bonus = self.bonus_scale * normalized_score
                self.score_sum += normalized_score
                self.score_buffer.append(normalized_score)

                if normalized_score > 0.99:
                    self.best_action_count += 1

            except Exception:
                bonus = 0.0

        # Execute the action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add expert bonus to reward
        reward += bonus

        # Adaptive decay
        self.steps += 1
        competency = self._compute_competency()

        if self.steps > self.warmup_steps and competency is not None:
            if self.bonus_scale > self.graduation_threshold:
                exponent = 1.0 + self.competency_multiplier * competency
                self.last_effective_decay = self.base_decay_rate ** exponent
                self.bonus_scale = max(
                    self.min_scale, self.bonus_scale * self.last_effective_decay
                )
            else:
                self.bonus_scale = self.min_scale

        # Info dict for logging
        info["value_bonus"] = bonus
        info["value_bonus_scale"] = self.bonus_scale
        if normalized_score is not None:
            info["value_normalized_score"] = normalized_score
        if self.total_decisions > 0:
            info["value_avg_score"] = self.score_sum / self.total_decisions
            info["value_best_action_rate"] = (
                self.best_action_count / self.total_decisions
            )

        # Curriculum-specific metrics
        info["curriculum_bonus_scale"] = self.bonus_scale
        info["curriculum_competency"] = competency if competency is not None else 0.0
        info["curriculum_progress"] = 1.0 - (self.bonus_scale / self.initial_scale)
        info["curriculum_effective_decay"] = self.last_effective_decay
        phase = self._get_phase(competency)
        info["curriculum_phase"] = phase
        info["curriculum_phase_num"] = (
            0.0 if phase == "warmup" else 1.0 if phase == "adaptive" else 2.0
        )

        return obs, reward, terminated, truncated, info

    def _compute_competency(self):
        """Rolling mean of normalized_score from buffer."""
        if len(self.score_buffer) < self.min_samples:
            return None
        return sum(self.score_buffer) / len(self.score_buffer)

    def _get_phase(self, competency):
        """Return current curriculum phase."""
        if self.steps <= self.warmup_steps:
            return "warmup"
        if self.bonus_scale <= self.graduation_threshold:
            return "graduated"
        return "adaptive"

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
