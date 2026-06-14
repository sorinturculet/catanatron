# reward_functions.py

import numpy as np
from catanatron.state_functions import get_actual_victory_points, player_key
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


def expansion_aware_rewards(game, p0_color, vps_to_win):
    """
    Like production_aware_rewards but with brick/wood production weighted 2×
    to encourage expansion (settlements + roads) alongside OWS (ore/wheat/sheep).

    Motivation: v19 accumulated 2.07 hidden dev-VP cards and only 1.00 extra
    settlements — a pure OWS/dev-card strategy. Weighting brick+wood production
    twice as much signals that board presence (expansion) is also valuable.

    Rewards:
    - VP delta:           +0.1 per VP gained / -0.1 per VP lost
    - OWS prod delta:     +0.02 per production point (ore/wheat/sheep)
    - Brick+Wood delta:   +0.04 per production point (2× weight)
    - Win bonus:          +1.0
    - Loss penalty:       -1.0
    """
    my_vps = get_actual_victory_points(game.state, p0_color)

    if not hasattr(game, '_prev_state_exp'):
        game._prev_state_exp = {}
    prev_vps   = game._prev_state_exp.get(f'{p0_color}_vps',  2)
    prev_ows   = game._prev_state_exp.get(f'{p0_color}_ows',  0)
    prev_bw    = game._prev_state_exp.get(f'{p0_color}_bw',   0)

    production_features = build_production_features(True)
    prod_sample = production_features(game, p0_color)

    current_ows = (
        prod_sample[f"EFFECTIVE_P0_ORE_PRODUCTION"]
        + prod_sample[f"EFFECTIVE_P0_WHEAT_PRODUCTION"]
        + prod_sample[f"EFFECTIVE_P0_SHEEP_PRODUCTION"]
    )
    current_bw = (
        prod_sample[f"EFFECTIVE_P0_BRICK_PRODUCTION"]
        + prod_sample[f"EFFECTIVE_P0_WOOD_PRODUCTION"]
    )

    game._prev_state_exp[f'{p0_color}_vps'] = my_vps
    game._prev_state_exp[f'{p0_color}_ows'] = current_ows
    game._prev_state_exp[f'{p0_color}_bw']  = current_bw

    vp_delta  = my_vps - prev_vps
    ows_delta = current_ows - prev_ows
    bw_delta  = current_bw  - prev_bw

    reward = vp_delta * 0.1 + ows_delta * 0.02 + bw_delta * 0.04

    winning_color = game.winning_color()
    if winning_color is not None:
        reward += 1.0 if p0_color == winning_color else -1.0

    return reward


def settlement_aware_rewards(game, p0_color, vps_to_win):
    """
    Extends expansion_aware_rewards with an explicit settlement-building bonus.

    v22 analysis: agent built only ~0.22 new settlements per game (starts with 2,
    ends with ~1.05 total on board). Pure OWS + dev-card strategy, no board presence.
    This adds +0.08 per settlement placed to directly incentivize expansion.

    Rewards:
    - VP delta:           +0.10 per VP gained / -0.10 per VP lost
    - OWS prod delta:     +0.02 per production point (ore/wheat/sheep)
    - Brick+Wood delta:   +0.04 per production point (2× weight)
    - Settlement placed:  +0.08 per settlement piece used
    - Win bonus:          +1.0
    - Loss penalty:       -1.0
    """
    my_vps = get_actual_victory_points(game.state, p0_color)

    # Get settlements available from game state (decreases when a piece is placed)
    pkey = player_key(game.state, p0_color)  # e.g. "P0" or "P1"
    settlements_left = game.state.player_state[f"{pkey}_SETTLEMENTS_AVAILABLE"]

    if not hasattr(game, '_prev_state_settle'):
        game._prev_state_settle = {}
    prev_vps          = game._prev_state_settle.get(f'{p0_color}_vps', 2)
    prev_ows          = game._prev_state_settle.get(f'{p0_color}_ows', 0)
    prev_bw           = game._prev_state_settle.get(f'{p0_color}_bw',  0)
    prev_settle_left  = game._prev_state_settle.get(f'{p0_color}_sl',  5)  # 5 pieces at game start

    production_features = build_production_features(True)
    prod_sample = production_features(game, p0_color)

    current_ows = (
        prod_sample["EFFECTIVE_P0_ORE_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_WHEAT_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_SHEEP_PRODUCTION"]
    )
    current_bw = (
        prod_sample["EFFECTIVE_P0_BRICK_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_WOOD_PRODUCTION"]
    )

    game._prev_state_settle[f'{p0_color}_vps'] = my_vps
    game._prev_state_settle[f'{p0_color}_ows'] = current_ows
    game._prev_state_settle[f'{p0_color}_bw']  = current_bw
    game._prev_state_settle[f'{p0_color}_sl']  = settlements_left

    vp_delta          = my_vps - prev_vps
    ows_delta         = current_ows - prev_ows
    bw_delta          = current_bw  - prev_bw
    settlements_placed = max(0, prev_settle_left - settlements_left)  # piece used = settlement placed

    reward = (
        vp_delta          * 0.10
        + ows_delta       * 0.02
        + bw_delta        * 0.04
        + settlements_placed * 0.08
    )

    winning_color = game.winning_color()
    if winning_color is not None:
        reward += 1.0 if p0_color == winning_color else -1.0

    return reward


def road_settlement_aware_rewards(game, p0_color, vps_to_win):
    """
    Extends settlement_aware_rewards with a road-building bonus.

    v23 analysis: agent has AVG ROAD 0.02 (almost never wins Longest Road).
    Adding +0.04 per road built to incentivize expansion alongside settlements.

    Rewards:
    - VP delta:           +0.10 per VP gained / -0.10 per VP lost
    - OWS prod delta:     +0.02 per production point (ore/wheat/sheep)
    - Brick+Wood delta:   +0.04 per production point (2× weight)
    - Settlement placed:  +0.10 per settlement piece used (bumped from 0.08)
    - Road built:         +0.04 per road piece used
    - Win bonus:          +1.0
    - Loss penalty:       -1.0
    """
    my_vps = get_actual_victory_points(game.state, p0_color)

    pkey = player_key(game.state, p0_color)
    settlements_left = game.state.player_state[f"{pkey}_SETTLEMENTS_AVAILABLE"]
    roads_left = game.state.player_state[f"{pkey}_ROADS_AVAILABLE"]

    if not hasattr(game, '_prev_state_rs'):
        game._prev_state_rs = {}
    prev_vps         = game._prev_state_rs.get(f'{p0_color}_vps', 2)
    prev_ows         = game._prev_state_rs.get(f'{p0_color}_ows', 0)
    prev_bw          = game._prev_state_rs.get(f'{p0_color}_bw',  0)
    prev_settle_left = game._prev_state_rs.get(f'{p0_color}_sl',  5)
    prev_roads_left  = game._prev_state_rs.get(f'{p0_color}_rl',  15)

    production_features = build_production_features(True)
    prod_sample = production_features(game, p0_color)

    current_ows = (
        prod_sample["EFFECTIVE_P0_ORE_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_WHEAT_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_SHEEP_PRODUCTION"]
    )
    current_bw = (
        prod_sample["EFFECTIVE_P0_BRICK_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_WOOD_PRODUCTION"]
    )

    game._prev_state_rs[f'{p0_color}_vps'] = my_vps
    game._prev_state_rs[f'{p0_color}_ows'] = current_ows
    game._prev_state_rs[f'{p0_color}_bw']  = current_bw
    game._prev_state_rs[f'{p0_color}_sl']  = settlements_left
    game._prev_state_rs[f'{p0_color}_rl']  = roads_left

    vp_delta          = my_vps - prev_vps
    ows_delta         = current_ows - prev_ows
    bw_delta          = current_bw  - prev_bw
    settlements_placed = max(0, prev_settle_left - settlements_left)
    roads_built        = max(0, prev_roads_left  - roads_left)

    reward = (
        vp_delta           * 0.10
        + ows_delta        * 0.02
        + bw_delta         * 0.04
        + settlements_placed * 0.10
        + roads_built      * 0.04
    )

    winning_color = game.winning_color()
    if winning_color is not None:
        reward += 1.0 if p0_color == winning_color else -1.0

    return reward


def city_road_settlement_aware_rewards(game, p0_color, vps_to_win):
    """
    Extends road_settlement_aware_rewards with a city upgrade bonus.

    v26 analysis: agent builds 1.27 cities vs VFP's 2.04 — undervalues upgrades.
    Adding +0.04 per city upgrade. The production increase from doubling hex output
    is already captured by OWS/BW deltas, so we keep the city bonus small and flat.

    Rewards:
    - VP delta:           +0.10 per VP gained / -0.10 per VP lost
    - OWS prod delta:     +0.02 per production point (ore/wheat/sheep)
    - Brick+Wood delta:   +0.04 per production point (2× weight)
    - Settlement placed:  +0.10 per settlement piece used
    - Road built:         +0.04 per road piece used
    - City upgrade:       +0.04 per city piece used
    - Win bonus:          +1.0
    - Loss penalty:       -1.0
    """
    my_vps = get_actual_victory_points(game.state, p0_color)

    pkey = player_key(game.state, p0_color)
    settlements_left = game.state.player_state[f"{pkey}_SETTLEMENTS_AVAILABLE"]
    roads_left = game.state.player_state[f"{pkey}_ROADS_AVAILABLE"]
    cities_left = game.state.player_state[f"{pkey}_CITIES_AVAILABLE"]

    if not hasattr(game, '_prev_state_crs'):
        game._prev_state_crs = {}
    prev_vps         = game._prev_state_crs.get(f'{p0_color}_vps', 2)
    prev_ows         = game._prev_state_crs.get(f'{p0_color}_ows', 0)
    prev_bw          = game._prev_state_crs.get(f'{p0_color}_bw',  0)
    prev_settle_left = game._prev_state_crs.get(f'{p0_color}_sl',  5)
    prev_roads_left  = game._prev_state_crs.get(f'{p0_color}_rl',  15)
    prev_cities_left = game._prev_state_crs.get(f'{p0_color}_cl',  4)

    production_features = build_production_features(True)
    prod_sample = production_features(game, p0_color)

    current_ows = (
        prod_sample["EFFECTIVE_P0_ORE_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_WHEAT_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_SHEEP_PRODUCTION"]
    )
    current_bw = (
        prod_sample["EFFECTIVE_P0_BRICK_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_WOOD_PRODUCTION"]
    )

    game._prev_state_crs[f'{p0_color}_vps'] = my_vps
    game._prev_state_crs[f'{p0_color}_ows'] = current_ows
    game._prev_state_crs[f'{p0_color}_bw']  = current_bw
    game._prev_state_crs[f'{p0_color}_sl']  = settlements_left
    game._prev_state_crs[f'{p0_color}_rl']  = roads_left
    game._prev_state_crs[f'{p0_color}_cl']  = cities_left

    vp_delta           = my_vps - prev_vps
    ows_delta          = current_ows - prev_ows
    bw_delta           = current_bw  - prev_bw
    settlements_placed = max(0, prev_settle_left - settlements_left)
    roads_built        = max(0, prev_roads_left  - roads_left)
    cities_built       = max(0, prev_cities_left - cities_left)

    reward = (
        vp_delta            * 0.10
        + ows_delta         * 0.02
        + bw_delta          * 0.04
        + settlements_placed * 0.10
        + roads_built       * 0.04
        + cities_built      * 0.04
    )

    winning_color = game.winning_color()
    if winning_color is not None:
        reward += 1.0 if p0_color == winning_color else -1.0

    return reward


def expansion_heavy_rewards(game, p0_color, vps_to_win):
    """
    v28: Heavily boosted expansion rewards to fix under-expansion problem.

    v27 analysis: 31.6% WR but only 1.15 settlements, 0.12 roads vs VFP's
    3.22 settlements, 0.84 roads. The model wins through dev cards (1.80 dev VP)
    but barely builds on the board. The previous settlement (+0.10) and road
    (+0.04) bonuses weren't strong enough to shift behavior.

    v28 doubles down on expansion:
    - Settlement placed:  +0.25 (was +0.10, 2.5× boost)
    - Road built:         +0.12 (was +0.04, 3× boost)
    - City upgrade:       +0.08 (was +0.04, 2× boost)
    - VP delta:           +0.10 per VP gained / -0.10 per VP lost
    - OWS prod delta:     +0.02 per production point (ore/wheat/sheep)
    - Brick+Wood delta:   +0.04 per production point (2× weight)
    - Win bonus:          +1.0
    - Loss penalty:       -1.0

    Combined with gamma=0.997 (up from 0.99) to value long-term expansion payoff.
    """
    my_vps = get_actual_victory_points(game.state, p0_color)

    pkey = player_key(game.state, p0_color)
    settlements_left = game.state.player_state[f"{pkey}_SETTLEMENTS_AVAILABLE"]
    roads_left = game.state.player_state[f"{pkey}_ROADS_AVAILABLE"]
    cities_left = game.state.player_state[f"{pkey}_CITIES_AVAILABLE"]

    if not hasattr(game, '_prev_state_exp_heavy'):
        game._prev_state_exp_heavy = {}
    prev_vps         = game._prev_state_exp_heavy.get(f'{p0_color}_vps', 2)
    prev_ows         = game._prev_state_exp_heavy.get(f'{p0_color}_ows', 0)
    prev_bw          = game._prev_state_exp_heavy.get(f'{p0_color}_bw',  0)
    prev_settle_left = game._prev_state_exp_heavy.get(f'{p0_color}_sl',  5)
    prev_roads_left  = game._prev_state_exp_heavy.get(f'{p0_color}_rl',  15)
    prev_cities_left = game._prev_state_exp_heavy.get(f'{p0_color}_cl',  4)

    production_features = build_production_features(True)
    prod_sample = production_features(game, p0_color)

    current_ows = (
        prod_sample["EFFECTIVE_P0_ORE_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_WHEAT_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_SHEEP_PRODUCTION"]
    )
    current_bw = (
        prod_sample["EFFECTIVE_P0_BRICK_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_WOOD_PRODUCTION"]
    )

    game._prev_state_exp_heavy[f'{p0_color}_vps'] = my_vps
    game._prev_state_exp_heavy[f'{p0_color}_ows'] = current_ows
    game._prev_state_exp_heavy[f'{p0_color}_bw']  = current_bw
    game._prev_state_exp_heavy[f'{p0_color}_sl']  = settlements_left
    game._prev_state_exp_heavy[f'{p0_color}_rl']  = roads_left
    game._prev_state_exp_heavy[f'{p0_color}_cl']  = cities_left

    vp_delta           = my_vps - prev_vps
    ows_delta          = current_ows - prev_ows
    bw_delta           = current_bw  - prev_bw
    settlements_placed = max(0, prev_settle_left - settlements_left)
    roads_built        = max(0, prev_roads_left  - roads_left)
    cities_built       = max(0, prev_cities_left - cities_left)

    reward = (
        vp_delta            * 0.10
        + ows_delta         * 0.02
        + bw_delta          * 0.04
        + settlements_placed * 0.25
        + roads_built       * 0.12
        + cities_built      * 0.08
    )

    winning_color = game.winning_color()
    if winning_color is not None:
        reward += 1.0 if p0_color == winning_color else -1.0

    return reward


def expansion_heavy_hand_penalty_rewards(game, p0_color, vps_to_win):
    """
    v29: Balanced expansion rewards + unified production + hand penalty.

    v28 hit 34% with expansion-heavy rewards. Still under-expanding (2.06 settles
    vs VFP's 3.17). v29 simplifies and rebalances:
    1. Unified production delta (no OWS/BW split) — rewards WHERE you build
    2. Equal settle/city bonus (+0.20) — cities are at least as valuable
    3. Hand penalty: -0.01 per card over 7 (spend, don't hoard)

    Rewards:
    - VP delta:           +0.10 per VP gained / -0.10 per VP lost
    - Production delta:   +0.10 per production point (unified, any resource)
    - Settlement placed:  +0.20 per settlement
    - Road built:         +0.10 per road
    - City upgrade:       +0.20 per city
    - Hand penalty:       -0.01 per card over 7
    - Win bonus:          +1.0
    - Loss penalty:       -1.0
    """
    my_vps = get_actual_victory_points(game.state, p0_color)

    pkey = player_key(game.state, p0_color)
    settlements_left = game.state.player_state[f"{pkey}_SETTLEMENTS_AVAILABLE"]
    roads_left = game.state.player_state[f"{pkey}_ROADS_AVAILABLE"]
    cities_left = game.state.player_state[f"{pkey}_CITIES_AVAILABLE"]

    # Count cards in hand
    num_cards = sum(
        game.state.player_state[f"{pkey}_{r}_IN_HAND"]
        for r in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
    )

    if not hasattr(game, '_prev_state_ehp'):
        game._prev_state_ehp = {}
    prev_vps         = game._prev_state_ehp.get(f'{p0_color}_vps', 2)
    prev_prod        = game._prev_state_ehp.get(f'{p0_color}_prod', 0)
    prev_settle_left = game._prev_state_ehp.get(f'{p0_color}_sl',  5)
    prev_roads_left  = game._prev_state_ehp.get(f'{p0_color}_rl',  15)
    prev_cities_left = game._prev_state_ehp.get(f'{p0_color}_cl',  4)

    production_features = build_production_features(True)
    prod_sample = production_features(game, p0_color)

    current_prod = (
        prod_sample["EFFECTIVE_P0_ORE_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_WHEAT_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_SHEEP_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_BRICK_PRODUCTION"]
        + prod_sample["EFFECTIVE_P0_WOOD_PRODUCTION"]
    )

    game._prev_state_ehp[f'{p0_color}_vps']  = my_vps
    game._prev_state_ehp[f'{p0_color}_prod'] = current_prod
    game._prev_state_ehp[f'{p0_color}_sl']   = settlements_left
    game._prev_state_ehp[f'{p0_color}_rl']   = roads_left
    game._prev_state_ehp[f'{p0_color}_cl']   = cities_left

    vp_delta           = my_vps - prev_vps
    prod_delta         = current_prod - prev_prod
    settlements_placed = max(0, prev_settle_left - settlements_left)
    roads_built        = max(0, prev_roads_left  - roads_left)
    cities_built       = max(0, prev_cities_left - cities_left)

    # Hand penalty: -0.01 per card over 7
    hand_penalty = max(0, num_cards - 7) * -0.01

    reward = (
        vp_delta            * 0.10
        + prod_delta        * 0.10
        + settlements_placed * 0.20
        + roads_built       * 0.10
        + cities_built      * 0.20
        + hand_penalty
    )

    winning_color = game.winning_color()
    if winning_color is not None:
        reward += 1.0 if p0_color == winning_color else -1.0

    return reward


def pbrs_rewards(game, p0_color, vps_to_win):
    """
    v33: Potential-Based Reward Shaping (Ng et al. 1999).

    Uses VFP's heuristic components as potential functions, each normalized
    independently so ALL signals contribute (not just VP).

    Φ(s) is decomposed into weighted components:
      - VP:                 0.10 per VP gained (same scale as hand-crafted)
      - Production:         0.03 per production point gained
      - Enemy production:  -0.03 per enemy production point gained
      - Reachable prod:     0.02 per reachable production point
      - Buildable nodes:    0.01 per buildable node
      - Hand synergy:       0.02 per synergy point (0-1 range)
      - Longest road:       0.01 per road length
      - Dev cards:          0.01 per dev card
      - Army:               0.01 per knight played
      - Tile diversity:     0.005 per unique tile

    F(s, s') = γΦ(s') - Φ(s) for each component, summed.

    Terminal:
      - Win:  +1.0
      - Loss: -1.0
    """
    from catanatron_experimental.machine_learning.players.value import base_fn
    from catanatron.state_functions import (
        get_longest_road_length,
        get_played_dev_cards,
        player_key,
        player_num_dev_cards,
        player_num_resource_cards,
    )
    from catanatron.models.enums import RESOURCES, SETTLEMENT, CITY
    from catanatron_gym.features import (
        build_production_features,
        reachability_features,
        resource_hand_features,
    )

    gamma = 0.997

    # ─── Compute current feature values (same as base_fn internals) ───
    production_features_fn = build_production_features(True)
    our_prod_sample = production_features_fn(game, p0_color)
    enemy_colors = [c for c in game.state.colors if c != p0_color]
    enemy_prod_sample = production_features_fn(game, enemy_colors[0]) if enemy_colors else {}

    our_production = value_production(our_prod_sample, "P0")
    enemy_production = value_production(enemy_prod_sample, "P0", False) if enemy_prod_sample else 0

    key = player_key(game.state, p0_color)
    vp = game.state.player_state[f"{key}_VICTORY_POINTS"]
    longest_road = get_longest_road_length(game.state, p0_color)

    reachability_sample = reachability_features(game, p0_color, 2)
    reachable_prod_1 = sum(
        reachability_sample.get(f"P0_1_ROAD_REACHABLE_{r}", 0) for r in RESOURCES
    )

    hand_sample = resource_hand_features(game, p0_color)
    distance_to_city = (
        max(2 - hand_sample.get("P0_WHEAT_IN_HAND", 0), 0)
        + max(3 - hand_sample.get("P0_ORE_IN_HAND", 0), 0)
    ) / 5.0
    distance_to_settle = (
        max(1 - hand_sample.get("P0_WHEAT_IN_HAND", 0), 0)
        + max(1 - hand_sample.get("P0_SHEEP_IN_HAND", 0), 0)
        + max(1 - hand_sample.get("P0_BRICK_IN_HAND", 0), 0)
        + max(1 - hand_sample.get("P0_WOOD_IN_HAND", 0), 0)
    ) / 4.0
    hand_synergy = (2 - distance_to_city - distance_to_settle) / 2

    num_buildable = len(game.state.board.buildable_node_ids(p0_color))
    num_dev_cards = player_num_dev_cards(game.state, p0_color)
    army = get_played_dev_cards(game.state, p0_color, "KNIGHT")

    buildings = game.state.buildings_by_color[p0_color]
    owned_nodes = buildings[SETTLEMENT] + buildings[CITY]
    owned_tiles = set()
    for n in owned_nodes:
        owned_tiles.update(game.state.board.map.adjacent_tiles[n])
    num_tiles = len(owned_tiles)

    # ─── Build current potential as dict of normalized components ───
    current = {
        "vp":           vp * 0.10,
        "production":   our_production * 0.03,
        "enemy_prod":   enemy_production * -0.03,
        "reachable":    reachable_prod_1 * 0.02,
        "buildable":    num_buildable * 0.01,
        "hand_synergy": hand_synergy * 0.02,
        "longest_road": longest_road * 0.01,
        "dev_cards":    num_dev_cards * 0.01,
        "army":         army * 0.01,
        "tiles":        num_tiles * 0.005,
    }

    # ─── Previous potential ───
    if not hasattr(game, '_pbrs_prev'):
        game._pbrs_prev = {}
    prev = game._pbrs_prev.get(p0_color, {k: 0.0 for k in current})

    # ─── PBRS: sum of γΦ_i(s') - Φ_i(s) for each component ───
    shaping_reward = sum(
        gamma * current[k] - prev[k] for k in current
    )

    game._pbrs_prev[p0_color] = current

    # Terminal bonus
    terminal_reward = 0.0
    winning_color = game.winning_color()
    if winning_color is not None:
        terminal_reward = 1.0 if p0_color == winning_color else -1.0

    return shaping_reward + terminal_reward


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
