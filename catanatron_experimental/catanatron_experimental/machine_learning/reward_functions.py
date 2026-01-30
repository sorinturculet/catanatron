# reward_functions.py

import numpy as np
from catanatron.state_functions import get_actual_victory_points
from catanatron_gym.features import build_production_features
from catanatron.models.enums import RESOURCES

# Production calculation (same as ValueFunctionPlayer uses)
TRANSLATE_VARIETY = 4  # each new resource type is like 4 production points


def value_production(sample, player_name="P0", include_variety=True):
    """Calculate production value from feature sample."""
    proba_point = 2.778 / 100
    features = [
        f"EFFECTIVE_{player_name}_WHEAT_PRODUCTION",
        f"EFFECTIVE_{player_name}_ORE_PRODUCTION",
        f"EFFECTIVE_{player_name}_SHEEP_PRODUCTION",
        f"EFFECTIVE_{player_name}_WOOD_PRODUCTION",
        f"EFFECTIVE_{player_name}_BRICK_PRODUCTION",
    ]
    prod_sum = sum([sample[f] for f in features])
    prod_variety = (
        sum([sample[f] != 0 for f in features]) * TRANSLATE_VARIETY * proba_point
    )
    return prod_sum + (0 if not include_variety else prod_variety)


def partial_rewards(game, p0_color, vps_to_win):
    """
    Calculate the partial rewards for the game.

    Args:
        game: The game instance.
        p0_color: The color representing the player's position.
        vps_to_win: The victory points required to win the game.

    Returns:
        A float representing the partial reward.
    """
    winning_color = game.winning_color()
    if winning_color is None:
        return 0

    total = 0
    if p0_color == winning_color:
        total += 0.20
    else:
        total -= 0.20
    enemy_vps = [
        get_actual_victory_points(game.state, color)
        for color in game.state.colors
        if color != p0_color
    ]
    enemy_avg_vp = sum(enemy_vps) / len(enemy_vps)
    my_vps = get_actual_victory_points(game.state, p0_color)
    vp_diff = (my_vps - enemy_avg_vp) / (vps_to_win - 1)

    total += 0.80 * vp_diff
    print(f"my_vps = {my_vps} enemy_avg_vp = {enemy_avg_vp} partial_rewards = {total}")
    return total


def dense_vp_rewards(game, p0_color, vps_to_win):
    """
    Dense reward function that provides feedback on every step based on VP changes.

    Rewards:
    - VP gain: +0.1 per VP gained
    - VP loss: -0.1 per VP lost (can happen if opponent takes longest road/army)
    - Win bonus: +1.0
    - Loss penalty: -1.0

    Args:
        game: The game instance.
        p0_color: The color representing the player's position.
        vps_to_win: The victory points required to win the game.

    Returns:
        A float representing the reward for this step.
    """
    my_vps = get_actual_victory_points(game.state, p0_color)

    # Track previous VP using game object (persists across steps)
    prev_vps = getattr(game, '_prev_vps', {}).get(p0_color, 2)  # Default 2 for starting settlements
    if not hasattr(game, '_prev_vps'):
        game._prev_vps = {}
    game._prev_vps[p0_color] = my_vps

    # Calculate VP delta reward
    vp_delta = my_vps - prev_vps
    reward = vp_delta * 0.1  # Scale factor: +0.1 per VP

    # Terminal bonus/penalty
    winning_color = game.winning_color()
    if winning_color is not None:
        if p0_color == winning_color:
            reward += 1.0  # Win bonus
        else:
            reward -= 1.0  # Loss penalty

    return reward


def production_aware_rewards(game, p0_color, vps_to_win):
    """
    Dense rewards with production signal (what ValueFunctionPlayer values).

    This reward function adds a production component to help the agent learn
    that building on high-probability tiles is valuable, even before it
    translates directly to victory points.

    Rewards:
    - VP delta: +0.1 per VP gained, -0.1 per VP lost
    - Production delta: +0.02 per production point gained
    - Win bonus: +1.0
    - Loss penalty: -1.0

    Args:
        game: The game instance.
        p0_color: The color representing the player's position.
        vps_to_win: The victory points required to win the game.

    Returns:
        A float representing the reward for this step.
    """
    my_vps = get_actual_victory_points(game.state, p0_color)

    # Track previous state using game object (persists across steps)
    if not hasattr(game, '_prev_state'):
        game._prev_state = {}
    prev_vps = game._prev_state.get(f'{p0_color}_vps', 2)  # Default 2 for starting settlements
    prev_prod = game._prev_state.get(f'{p0_color}_prod', 0)

    # Calculate current production value
    production_features = build_production_features(True)
    prod_sample = production_features(game, p0_color)
    current_prod = value_production(prod_sample, "P0")

    # Update tracking
    game._prev_state[f'{p0_color}_vps'] = my_vps
    game._prev_state[f'{p0_color}_prod'] = current_prod

    # Calculate rewards
    vp_delta = my_vps - prev_vps
    prod_delta = current_prod - prev_prod

    # VP weighted 5x more than production (0.1 vs 0.02)
    reward = vp_delta * 0.1 + prod_delta * 0.02

    # Terminal bonus/penalty
    winning_color = game.winning_color()
    if winning_color is not None:
        if p0_color == winning_color:
            reward += 1.0  # Win bonus
        else:
            reward -= 1.0  # Loss penalty

    return reward


def mask_fn(env) -> np.ndarray:
    """
    Generates a boolean mask of valid actions for the environment.

    Args:
        env: The environment instance.

    Returns:
        A numpy array of booleans indicating valid actions.
    """
    valid_actions = env.unwrapped.get_valid_actions()
    mask = np.zeros(env.action_space.n, dtype=bool)
    mask[valid_actions] = True
    return mask
