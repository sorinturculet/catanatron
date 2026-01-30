"""
Expert-Guided Wrapper for Online Imitation Learning.

This wrapper adds a reward bonus when the agent's action matches what
an expert (e.g., ValueFunctionPlayer) would have chosen. This provides
implicit imitation learning signal alongside the standard RL rewards.
"""

import gymnasium as gym
from catanatron_gym.envs.catanatron_env import to_action_space


class ExpertGuidedWrapper(gym.Wrapper):
    """
    Gym wrapper that adds reward bonus for matching expert decisions.

    Args:
        env: The base Catan environment
        expert_player: Expert player to query (e.g., ValueFunctionPlayer)
        agreement_bonus: Reward bonus when agent matches expert (default: 0.1)
        decay_start_step: Step at which to start decaying bonus (default: 5M)
        decay_rate: Exponential decay rate after decay_start_step (default: 0.99999)
        min_bonus: Minimum bonus after decay (default: 0.02)
    """

    def __init__(
        self,
        env,
        expert_player,
        agreement_bonus=0.1,
        decay_start_step=5_000_000,
        decay_rate=0.99999,
        min_bonus=0.02,
    ):
        super().__init__(env)
        self.expert = expert_player
        self.agreement_bonus = agreement_bonus
        self.initial_bonus = agreement_bonus
        self.decay_start_step = decay_start_step
        self.decay_rate = decay_rate
        self.min_bonus = min_bonus

        # Statistics
        self.steps = 0
        self.agreements = 0
        self.total_decisions = 0
        self.recent_agreements = []  # Rolling window for recent agreement rate
        self.window_size = 1000

    def step(self, action):
        """Execute action and add expert agreement bonus to reward."""
        # Get expert's preferred action BEFORE executing agent's action
        game = self.env.unwrapped.game
        playable_actions = game.state.playable_actions

        agreed = False
        expert_action_idx = None

        if len(playable_actions) > 1:
            # Non-trivial decision - query expert
            try:
                expert_action = self.expert.decide(game, playable_actions)
                expert_action_idx = to_action_space(expert_action)
                agreed = (action == expert_action_idx)

                # Update statistics
                self.total_decisions += 1
                if agreed:
                    self.agreements += 1

                # Update rolling window
                self.recent_agreements.append(1 if agreed else 0)
                if len(self.recent_agreements) > self.window_size:
                    self.recent_agreements.pop(0)
            except Exception as e:
                # If expert fails, don't penalize agent
                agreed = True
        else:
            # Trivial decision (only 1 option) - always agree
            agreed = True

        # Execute agent's action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add agreement bonus
        if agreed:
            reward += self.agreement_bonus

        # Update step counter and decay bonus after decay_start_step
        self.steps += 1
        if self.steps > self.decay_start_step:
            self.agreement_bonus = max(
                self.min_bonus,
                self.agreement_bonus * self.decay_rate
            )

        # Add statistics to info dict for logging
        info['expert_agreed'] = agreed
        info['expert_action'] = expert_action_idx
        info['agreement_bonus'] = self.agreement_bonus

        if self.total_decisions > 0:
            info['agreement_rate_total'] = self.agreements / self.total_decisions
        else:
            info['agreement_rate_total'] = 0.0

        if len(self.recent_agreements) > 0:
            info['agreement_rate_recent'] = sum(self.recent_agreements) / len(self.recent_agreements)
        else:
            info['agreement_rate_recent'] = 0.0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment. Statistics persist across episodes."""
        return self.env.reset(**kwargs)

    def get_stats(self):
        """Return current statistics."""
        return {
            'total_steps': self.steps,
            'total_decisions': self.total_decisions,
            'total_agreements': self.agreements,
            'agreement_rate': self.agreements / max(1, self.total_decisions),
            'current_bonus': self.agreement_bonus,
        }
