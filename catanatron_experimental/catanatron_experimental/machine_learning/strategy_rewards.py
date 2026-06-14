"""
Strategy-specific reward components for hierarchical PPO (v14).

Each function computes a reward delta for progress toward a strategy goal.
These are combined with weights in the wrapper.
"""

from catanatron.state_functions import (
    get_player_buildings,
    get_longest_road_length,
    get_played_dev_cards,
    get_actual_victory_points,
    player_key,
)
from catanatron.models.enums import SETTLEMENT, CITY


def compute_city_progress(game, color, prev_state):
    """
    Reward for city-engine progress.

    Tracks:
    - Cities built (high value)
    """
    # Current state
    num_cities = len(get_player_buildings(game.state, color, CITY))

    # Previous state
    prev_cities = prev_state.get('cities', 0)

    # Deltas
    cities_delta = num_cities - prev_cities

    # Update prev_state for next call
    prev_state['cities'] = num_cities

    # Reward: cities are the main goal (0.5 per city built)
    return cities_delta * 0.5


def compute_expansion_progress(game, color, prev_state):
    """
    Reward for expansion progress.

    Tracks:
    - Settlements built (high value)
    - Roads built / longest road progress (medium value)
    """
    key = player_key(game.state, color)

    # Current state
    num_settlements = len(get_player_buildings(game.state, color, SETTLEMENT))
    road_length = get_longest_road_length(game.state, color)
    has_longest = game.state.player_state[f"{key}_HAS_ROAD"]

    # Previous state (start with 2 settlements from initial placement)
    prev_settlements = prev_state.get('settlements', 2)
    prev_road_length = prev_state.get('road_length', 0)
    prev_has_longest = prev_state.get('has_longest_road', False)

    # Deltas
    settlements_delta = num_settlements - prev_settlements
    road_delta = road_length - prev_road_length
    got_longest = has_longest and not prev_has_longest

    # Update prev_state
    prev_state['settlements'] = num_settlements
    prev_state['road_length'] = road_length
    prev_state['has_longest_road'] = has_longest

    # Reward
    reward = settlements_delta * 0.3  # 0.3 per settlement
    reward += road_delta * 0.05       # 0.05 per road segment
    if got_longest:
        reward += 0.5                 # 0.5 for getting longest road

    return reward


def compute_dev_progress(game, color, prev_state):
    """
    Reward for dev-card progress.

    Tracks:
    - Knights played (for largest army)
    - Getting largest army
    """
    key = player_key(game.state, color)

    # Current state
    knights_played = get_played_dev_cards(game.state, color, "KNIGHT")
    has_army = game.state.player_state[f"{key}_HAS_ARMY"]

    # Previous state
    prev_knights = prev_state.get('knights_played', 0)
    prev_has_army = prev_state.get('has_army', False)

    # Deltas
    knights_delta = knights_played - prev_knights
    got_army = has_army and not prev_has_army

    # Update prev_state
    prev_state['knights_played'] = knights_played
    prev_state['has_army'] = has_army

    # Reward
    reward = knights_delta * 0.15    # 0.15 per knight played
    if got_army:
        reward += 0.5                # 0.5 for getting largest army

    return reward
