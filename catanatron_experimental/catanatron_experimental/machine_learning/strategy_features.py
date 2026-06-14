"""
Strategy weight computation for hierarchical PPO (v14).

Computes weights based on player's production profile:
- city_weight: potential for city-engine strategy (ore + wheat)
- expansion_weight: potential for expansion strategy (wood + brick)
- dev_weight: potential for dev-card strategy (sheep + wheat + ore)

Weights are normalized to sum to 1.0, allowing hybrid strategies.
"""

from catanatron.state_functions import get_player_buildings
from catanatron.models.enums import SETTLEMENT, CITY, WOOD, BRICK, SHEEP, WHEAT, ORE
from catanatron_gym.features import get_node_production


def get_player_production(game, color):
    """
    Calculate effective production per resource for a player.

    Returns dict: {WOOD: float, BRICK: float, SHEEP: float, WHEAT: float, ORE: float}
    Production values are probabilities (e.g., 0.14 = 14% chance per roll)
    """
    board = game.state.board
    robber_coord = board.robber_coordinate
    robbed_nodes = set(board.map.tiles[robber_coord].nodes.values())

    production = {WOOD: 0.0, BRICK: 0.0, SHEEP: 0.0, WHEAT: 0.0, ORE: 0.0}

    # Settlements contribute 1x
    for node_id in get_player_buildings(game.state, color, SETTLEMENT):
        if node_id in robbed_nodes:
            continue
        for resource in production.keys():
            production[resource] += get_node_production(
                board.map, node_id, resource
            )

    # Cities contribute 2x
    for node_id in get_player_buildings(game.state, color, CITY):
        if node_id in robbed_nodes:
            continue
        for resource in production.keys():
            production[resource] += 2 * get_node_production(
                board.map, node_id, resource
            )

    return production


def compute_strategy_weights(game, color):
    """
    Compute normalized strategy weights based on production profile.

    Returns dict with keys: 'city', 'expansion', 'dev'
    Values sum to 1.0

    Heuristics:
    - City engine needs ore (3) + wheat (2) per city
    - Expansion needs wood + brick (1 each for road, +sheep+wheat for settlement)
    - Dev cards need sheep + wheat + ore (1 each)
    """
    prod = get_player_production(game, color)

    ore = prod[ORE]
    wheat = prod[WHEAT]
    wood = prod[WOOD]
    brick = prod[BRICK]
    sheep = prod[SHEEP]

    # Raw scores (higher = better fit for strategy)
    # City: rate-limited by min(ore/3, wheat/2), but also benefit from surplus
    city_score = min(ore, wheat * 1.5) * 2 + ore + wheat

    # Expansion: rate-limited by min(wood, brick), need both equally
    expansion_score = min(wood, brick) * 2 + wood + brick

    # Dev cards: rate-limited by min(sheep, wheat, ore)
    dev_score = min(sheep, wheat, ore) * 3 + sheep

    # Normalize to weights summing to 1
    total = city_score + expansion_score + dev_score + 1e-6

    return {
        'city': city_score / total,
        'expansion': expansion_score / total,
        'dev': dev_score / total,
    }


def compute_strategy_weights_from_sample(sample, player_prefix="P0"):
    """
    Compute strategy weights from a feature sample dict.
    Useful when you already have create_sample() output.
    """
    ore = sample.get(f"EFFECTIVE_{player_prefix}_ORE_PRODUCTION", 0)
    wheat = sample.get(f"EFFECTIVE_{player_prefix}_WHEAT_PRODUCTION", 0)
    wood = sample.get(f"EFFECTIVE_{player_prefix}_WOOD_PRODUCTION", 0)
    brick = sample.get(f"EFFECTIVE_{player_prefix}_BRICK_PRODUCTION", 0)
    sheep = sample.get(f"EFFECTIVE_{player_prefix}_SHEEP_PRODUCTION", 0)

    city_score = min(ore, wheat * 1.5) * 2 + ore + wheat
    expansion_score = min(wood, brick) * 2 + wood + brick
    dev_score = min(sheep, wheat, ore) * 3 + sheep

    total = city_score + expansion_score + dev_score + 1e-6

    return {
        'city': city_score / total,
        'expansion': expansion_score / total,
        'dev': dev_score / total,
    }
