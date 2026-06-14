"""
Strategy-Weighted Wrapper for Hierarchical PPO (v14).

This wrapper:
1. Computes strategy weights after initial placement (episode start)
2. Augments observation with strategy weights (3 extra features)
3. Applies strategy-weighted reward shaping throughout episode
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from catanatron import Color
from strategy_features import compute_strategy_weights
from strategy_rewards import (
    compute_city_progress,
    compute_expansion_progress,
    compute_dev_progress,
)


class StrategyWeightedWrapper(gym.Wrapper):
    """
    Wrapper that adds strategy-conditioning to the environment.

    Args:
        env: Base Catan environment (with ActionMasker already applied)
        player_color: Color of the agent
        strategy_bonus_scale: Multiplier for strategy-specific rewards (default 0.3)
        recompute_weights_every_n: Recompute weights every N steps (0 = only at start)
    """

    def __init__(
        self,
        env,
        player_color=Color.BLUE,
        strategy_bonus_scale=0.3,
        recompute_weights_every_n=0,
    ):
        super().__init__(env)
        self.player_color = player_color
        self.strategy_bonus_scale = strategy_bonus_scale
        self.recompute_every_n = recompute_weights_every_n

        # Strategy weights (set at episode start)
        self.strategy_weights = {'city': 0.33, 'expansion': 0.33, 'dev': 0.34}

        # State tracking for reward deltas
        self.prev_state = {}
        self.steps_in_episode = 0

        # Modify observation space to include strategy weights (3 extra features)
        original_space = env.observation_space
        if isinstance(original_space, spaces.Dict):
            # Mixed representation: add to numeric
            original_numeric = original_space['numeric']
            new_numeric_shape = (original_numeric.shape[0] + 3,)

            self.observation_space = spaces.Dict({
                'board': original_space['board'],
                'numeric': spaces.Box(
                    low=0,
                    high=max(float(original_numeric.high.max()), 1.0),
                    shape=new_numeric_shape,
                    dtype=np.float32
                )
            })
        else:
            # Vector representation: append 3 features
            new_shape = (original_space.shape[0] + 3,)
            self.observation_space = spaces.Box(
                low=0,
                high=max(float(original_space.high.max()), 1.0),
                shape=new_shape,
                dtype=np.float32
            )

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)

        # Handle both (obs,) and (obs, info) returns
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        # Reset state tracking
        self.prev_state = {}
        self.steps_in_episode = 0

        # Compute strategy weights based on initial placement
        game = self.env.unwrapped.game
        self.strategy_weights = compute_strategy_weights(game, self.player_color)

        # Augment observation with strategy weights
        obs = self._augment_observation(obs)

        # Log initial weights
        info['strategy_city_weight'] = self.strategy_weights['city']
        info['strategy_expansion_weight'] = self.strategy_weights['expansion']
        info['strategy_dev_weight'] = self.strategy_weights['dev']

        return obs, info

    def step(self, action):
        # Execute action
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.steps_in_episode += 1
        game = self.env.unwrapped.game

        # Optionally recompute weights (for dynamic adaptation)
        if self.recompute_every_n > 0 and self.steps_in_episode % self.recompute_every_n == 0:
            self.strategy_weights = compute_strategy_weights(game, self.player_color)

        # Compute strategy-specific reward components
        city_reward = compute_city_progress(game, self.player_color, self.prev_state)
        expansion_reward = compute_expansion_progress(game, self.player_color, self.prev_state)
        dev_reward = compute_dev_progress(game, self.player_color, self.prev_state)

        # Weight by strategy profile
        strategy_bonus = (
            self.strategy_weights['city'] * city_reward +
            self.strategy_weights['expansion'] * expansion_reward +
            self.strategy_weights['dev'] * dev_reward
        ) * self.strategy_bonus_scale

        # Add to base reward
        reward += strategy_bonus

        # Augment observation
        obs = self._augment_observation(obs)

        # Logging
        info['strategy_bonus'] = strategy_bonus
        info['strategy_city_reward'] = city_reward
        info['strategy_expansion_reward'] = expansion_reward
        info['strategy_dev_reward'] = dev_reward
        info['strategy_city_weight'] = self.strategy_weights['city']
        info['strategy_expansion_weight'] = self.strategy_weights['expansion']
        info['strategy_dev_weight'] = self.strategy_weights['dev']

        return obs, reward, terminated, truncated, info

    def _augment_observation(self, obs):
        """Add strategy weights to observation."""
        weights_array = np.array([
            self.strategy_weights['city'],
            self.strategy_weights['expansion'],
            self.strategy_weights['dev'],
        ], dtype=np.float32)

        if isinstance(obs, dict):
            # Mixed representation
            return {
                'board': obs['board'],
                'numeric': np.concatenate([obs['numeric'], weights_array])
            }
        else:
            # Vector representation
            return np.concatenate([obs, weights_array])
