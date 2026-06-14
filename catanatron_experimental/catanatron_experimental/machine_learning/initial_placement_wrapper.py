"""
Initial Placement Heuristic Wrapper for v24.

Intercepts BUILD_SETTLEMENT and BUILD_ROAD actions during the initial build phase
and replaces the PPO agent's choice with a crafted heuristic based on:
  - Total production value (sum of dice probabilities from adjacent hexes)
  - Resource diversity bonus (unique resource types)
  - Port proximity bonus (2:1 or 3:1 ports)
  - 2nd settlement complementarity (prioritize missing resources from 1st settlement)

The PPO agent still handles ALL mid-game decisions. This wrapper only overrides
the 2 settlement + 2 road placements at game start.
"""

import gymnasium as gym
import numpy as np
from collections import Counter

from catanatron.models.enums import ActionType, SETTLEMENT, RESOURCES
from catanatron_gym.envs.catanatron_env import to_action_space, ACTIONS_ARRAY


# Weights for the heuristic scoring
PRODUCTION_WEIGHT = 1.0       # Raw production value (probability sum)
DIVERSITY_WEIGHT = 0.04       # Bonus per unique resource type (like VFP's TRANSLATE_VARIETY)
PORT_2TO1_WEIGHT = 0.03       # Bonus if node is on a 2:1 port matching a produced resource
PORT_3TO1_WEIGHT = 0.015      # Bonus if node is on a 3:1 port
COMPLEMENTARITY_WEIGHT = 0.05  # Bonus per NEW resource type not covered by 1st settlement


def score_settlement_node(node_id, catan_map, first_settlement_resources=None):
    """
    Score a node for initial settlement placement.

    Args:
        node_id: The node to score.
        catan_map: The CatanMap object (game.state.board.map).
        first_settlement_resources: Set of resource types produced by 1st settlement
            (None for 1st settlement scoring).

    Returns:
        float: Heuristic score for this node.
    """
    # Get production counter: {WOOD: 0.083, WHEAT: 0.139, ...}
    production = catan_map.node_production.get(node_id, Counter())

    # 1. Total production value
    total_prod = sum(production.values())
    score = PRODUCTION_WEIGHT * total_prod

    # 2. Resource diversity: number of unique resource types
    unique_resources = set(r for r, v in production.items() if v > 0)
    num_unique = len(unique_resources)
    score += DIVERSITY_WEIGHT * num_unique

    # 3. Port bonus
    for resource, port_node_set in catan_map.port_nodes.items():
        if node_id in port_node_set:
            if resource is None:
                # 3:1 port
                score += PORT_3TO1_WEIGHT
            elif resource in unique_resources:
                # 2:1 port for a resource we actually produce here
                score += PORT_2TO1_WEIGHT
            else:
                # 2:1 port for a resource we don't produce — minor value
                score += PORT_3TO1_WEIGHT * 0.5

    # 4. Complementarity with 1st settlement (only for 2nd placement)
    if first_settlement_resources is not None:
        new_resources = unique_resources - first_settlement_resources
        score += COMPLEMENTARITY_WEIGHT * len(new_resources)

    return score


def pick_best_settlement(game, player_color):
    """
    Pick the best settlement node from valid options using the heuristic.

    Args:
        game: The Game instance.
        player_color: The color of the player placing the settlement.

    Returns:
        int: Action index (into ACTIONS_ARRAY) for the best BUILD_SETTLEMENT action.
    """
    playable_actions = game.state.playable_actions
    catan_map = game.state.board.map

    # Check if this is 1st or 2nd settlement
    existing_settlements = game.state.buildings_by_color[player_color][SETTLEMENT]
    first_settlement_resources = None

    if len(existing_settlements) >= 1:
        # 2nd settlement — compute resources from 1st
        first_node = existing_settlements[0]
        first_prod = catan_map.node_production.get(first_node, Counter())
        first_settlement_resources = set(r for r, v in first_prod.items() if v > 0)

    best_score = -float('inf')
    best_action_idx = None

    for action in playable_actions:
        if action.action_type != ActionType.BUILD_SETTLEMENT:
            continue
        node_id = action.value
        score = score_settlement_node(node_id, catan_map, first_settlement_resources)
        if score > best_score:
            best_score = score
            best_action_idx = to_action_space(action)

    return best_action_idx


def pick_best_road(game, player_color):
    """
    Pick the best initial road. Prefer roads that point toward
    high-production nodes (future expansion potential).

    Args:
        game: The Game instance.
        player_color: The color of the player placing the road.

    Returns:
        int: Action index for the best BUILD_ROAD action.
    """
    playable_actions = game.state.playable_actions
    catan_map = game.state.board.map

    # The last settlement placed
    last_settlement = game.state.buildings_by_color[player_color][SETTLEMENT][-1]

    best_score = -float('inf')
    best_action_idx = None

    for action in playable_actions:
        if action.action_type != ActionType.BUILD_ROAD:
            continue
        edge = action.value
        # The "other" node (not the settlement we just placed)
        other_node = edge[0] if edge[1] == last_settlement else edge[1]

        # Score the other node by its production potential
        production = catan_map.node_production.get(other_node, Counter())
        score = sum(production.values())

        if score > best_score:
            best_score = score
            best_action_idx = to_action_space(action)

    return best_action_idx


class InitialPlacementWrapper(gym.Wrapper):
    """
    Gym wrapper that overrides initial settlement and road placement
    with a crafted heuristic. The PPO agent handles all other decisions.
    """

    def __init__(self, env, player_color):
        super().__init__(env)
        self.player_color = player_color

    def step(self, action):
        game = self.env.unwrapped.game
        state = game.state

        # Only intercept during initial build phase for our player
        if state.is_initial_build_phase and state.current_color() == self.player_color:
            playable = state.playable_actions
            if playable and playable[0].action_type == ActionType.BUILD_SETTLEMENT:
                heuristic_action = pick_best_settlement(game, self.player_color)
                if heuristic_action is not None:
                    action = heuristic_action
            elif playable and playable[0].action_type == ActionType.BUILD_ROAD:
                heuristic_action = pick_best_road(game, self.player_color)
                if heuristic_action is not None:
                    action = heuristic_action

        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
