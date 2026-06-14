from typing import Iterable
import numpy as np
import os
from sb3_contrib import MaskablePPO

from catanatron.game import Game
from catanatron.models.actions import Action
from catanatron.models.player import Player
from catanatron_gym.envs.catanatron_env import from_action_space, to_action_space
from catanatron_gym.features import create_sample, get_feature_ordering
from catanatron_gym.board_tensor_features import (
    create_board_tensor,
    is_graph_feature,
)
from catanatron_experimental.machine_learning.custom_cnn import CustomCNN

# Import strategy features for v14+ models
try:
    from catanatron_experimental.machine_learning.strategy_features import compute_strategy_weights
    STRATEGY_FEATURES_AVAILABLE = True
except ImportError:
    STRATEGY_FEATURES_AVAILABLE = False


class PPOPlayer(Player):
    """
    Proximal Policy Optimization (PPO) reinforcement learning agent.
    """

    def __init__(self, color, model_path=None):
        super().__init__(color)
        self.model = None
        self.model_path = model_path
        self.numeric_features = None
        self.uses_strategy_features = False  # Track if model needs strategy features
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "..", "model.zip")
            self.model_path = model_path
        if model_path:
            self.load(model_path)

    def __getstate__(self):
        """Exclude unpicklable model from serialization."""
        state = self.__dict__.copy()
        state['model'] = None
        state['numeric_features'] = None
        return state

    def __setstate__(self, state):
        """Restore state without reloading model — lazy load on next decide()."""
        self.__dict__.update(state)

    def decide(self, game: Game, playable_actions: Iterable[Action]):
        if self.model is None:
            if self.model_path:
                self.load(self.model_path)
            else:
                raise ValueError("Model not loaded. Call load() first.")

        # Initialize numeric_features based on the current game
        if self.numeric_features is None:
            num_players = len(game.state.players)
            self.features = get_feature_ordering(num_players)
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]

        # Generate observation from the game state
        obs = self.generate_observation(game)

        # Generate action mask from playable actions
        action_mask = self.generate_action_mask(playable_actions)

        # Predict the action index
        action_index, _ = self.model.predict(
            obs, action_masks=action_mask, deterministic=True
        )

        # Map the action index to the actual Action
        try:
            selected_action = self.action_index_to_action(
                action_index, playable_actions
            )
            return selected_action
        except Exception as e:
            print(f"Error mapping action index to Action: {e}")
            # Default to the first playable action
            return list(playable_actions)[0]

    def generate_observation(self, game: Game):
        # Create the sample
        sample = create_sample(game, self.color)

        # Generate board tensor
        board_tensor = create_board_tensor(
            game, self.color, channels_first=True
        ).astype(np.float32)

        # Extract numeric features
        numeric = np.array(
            [float(sample[i]) for i in self.numeric_features], dtype=np.float32
        )

        # Add strategy features if model expects them (v14+)
        if self.uses_strategy_features and STRATEGY_FEATURES_AVAILABLE:
            strategy_weights = compute_strategy_weights(game, self.color)
            strategy_array = np.array([
                strategy_weights['city'],
                strategy_weights['expansion'],
                strategy_weights['dev']
            ], dtype=np.float32)
            numeric = np.concatenate([numeric, strategy_array])

        # Create the observation
        obs = {"board": board_tensor, "numeric": numeric}
        return obs

    def generate_action_mask(self, playable_actions: Iterable[Action]):
        action_mask = np.zeros(self.model.action_space.n, dtype=bool)
        for action in playable_actions:
            try:
                action_index = self.action_to_action_index(action)
                if (
                    action_index is not None
                    and 0 <= action_index < self.model.action_space.n
                ):
                    action_mask[action_index] = True
            except Exception as e:
                print(f"Error in action_to_action_index: {e}")
                continue
        return action_mask

    def action_to_action_index(self, action: Action):
        action_index = to_action_space(action)
        return action_index

    def action_index_to_action(
        self, action_index: int, playable_actions: Iterable[Action]
    ):
        action = from_action_space(action_index, playable_actions)
        if action in playable_actions:
            return action
        else:
            raise ValueError(f"Action {action} not in playable actions.")

    def load(self, path):
        custom_objects = {"features_extractor_class": CustomCNN}

        # Try loading as a standard model first (v21 and earlier non-MoE models).
        # If that fails, retry as MoE model (v20).
        try:
            self.model = MaskablePPO.load(path, custom_objects=custom_objects)
        except Exception:
            try:
                from catanatron_experimental.machine_learning.moe_policy import MoEMaskablePolicy
                custom_objects["policy_class"] = MoEMaskablePolicy
                self.model = MaskablePPO.load(path, custom_objects=custom_objects)
            except ImportError:
                raise RuntimeError("Failed to load model: not a standard model and moe_policy not found.")

        # Validate observation space matches current features.py
        # Get expected numeric feature count from model
        model_obs_space = self.model.observation_space
        if hasattr(model_obs_space, 'spaces') and 'numeric' in model_obs_space.spaces:
            model_numeric_shape = model_obs_space.spaces['numeric'].shape[0]
        else:
            model_numeric_shape = model_obs_space.shape[0]

        # Get current feature count from features.py
        features = get_feature_ordering(num_players=2)  # 2-player game
        current_numeric = [f for f in features if not is_graph_feature(f)]
        current_numeric_count = len(current_numeric)

        # Check if model uses strategy features (v14+)
        # Strategy features add exactly 3 features: city_weight, expansion_weight, dev_weight
        if model_numeric_shape == current_numeric_count + 3:
            if STRATEGY_FEATURES_AVAILABLE:
                self.uses_strategy_features = True
                print(f"Detected v14+ model with strategy features ({model_numeric_shape} features)")
            else:
                raise ValueError(
                    f"Model expects strategy features but strategy_features.py not found.\n"
                    f"Model expects {model_numeric_shape} features, base is {current_numeric_count}."
                )
        elif model_numeric_shape != current_numeric_count:
            raise ValueError(
                f"\n{'='*70}\n"
                f"OBSERVATION SPACE MISMATCH!\n"
                f"{'='*70}\n"
                f"Model expects: {model_numeric_shape} numeric features\n"
                f"features.py provides: {current_numeric_count} numeric features\n"
                f"\n"
                f"This means the model was trained with a DIFFERENT features.py.\n"
                f"\n"
                f"TO FIX: Delete old model and retrain with current features.py:\n"
                f"   rm model_v11.zip\n"
                f"   python -m catanatron_experimental.machine_learning.train_ppo_agent_11\n"
                f"{'='*70}"
            )
