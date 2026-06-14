"""
Diverse VFP Opponent Pool for v37.

Instead of NN-based self-play (which runs at 6 FPS due to CPU inference in
each SubprocVecEnv worker), this module provides opponent diversity via
multiple ValueFunctionPlayer variants with different weight profiles.

All opponents are pure heuristic — zero NN overhead, same ~650 FPS as
training against a single VFP.

Paper backing:
  - Bansal et al. 2018 ("Emergent Complexity via Multi-Agent Competition"):
    opponent diversity prevents strategy collapse.
  - Parker-Holder et al. 2020 ("Effective Diversity in Population Based RL"):
    diverse opponents improve sample efficiency and generalization.

Variants:
  1. VFP-default:    Original DEFAULT_WEIGHTS (the eval opponent)
  2. VFP-expansion:  Boosted buildable_nodes + reachable_production
  3. VFP-devcard:    Boosted hand_devs + army_size
  4. VFP-contender:  CONTENDER_WEIGHTS (existing alternative tuning)
  5. VFP-noisy:      DEFAULT_WEIGHTS + epsilon=0.15 (15% random actions)
"""

import random
from typing import Iterable

from catanatron.game import Game
from catanatron.models.actions import Action
from catanatron.models.player import Player
from catanatron_experimental.machine_learning.players.value import (
    ValueFunctionPlayer,
    DEFAULT_WEIGHTS,
    CONTENDER_WEIGHTS,
)

# --- Weight variants ---

VFP_EXPANSION_WEIGHTS = {
    **DEFAULT_WEIGHTS,
    "buildable_nodes": DEFAULT_WEIGHTS["buildable_nodes"] * 10,  # 1e3 -> 1e4
    "reachable_production_1": DEFAULT_WEIGHTS["reachable_production_1"] * 5,  # 1e4 -> 5e4
    "longest_road": DEFAULT_WEIGHTS["longest_road"] * 5,  # 10 -> 50
}

VFP_DEVCARD_WEIGHTS = {
    **DEFAULT_WEIGHTS,
    "hand_devs": DEFAULT_WEIGHTS["hand_devs"] * 10,  # 10 -> 100
    "army_size": DEFAULT_WEIGHTS["army_size"] * 10,  # 10.1 -> 101
}

# (tag, params, epsilon)
VFP_VARIANTS = [
    ("default",   DEFAULT_WEIGHTS,       None),
    ("expansion", VFP_EXPANSION_WEIGHTS, None),
    ("devcard",   VFP_DEVCARD_WEIGHTS,   None),
    ("contender", CONTENDER_WEIGHTS,     None),
    ("noisy",     DEFAULT_WEIGHTS,       0.15),
]


class DiverseVFPOpponent(Player):
    """Player that samples a VFP variant per episode (on reset_state).

    All backends are heuristic — no NN inference, no GPU, no model loading.
    Each variant is lazily constructed and cached.
    """

    def __init__(self, color):
        super().__init__(color)
        self._backends = {}
        self._active = None
        self._active_tag = None
        self._rng = random.Random()

    def _get_backend(self, tag, params, epsilon):
        if tag not in self._backends:
            # Use "C" (contender path) for custom weights — base_fn path
            # ignores the params arg and always uses DEFAULT_WEIGHTS.
            # contender_fn(params) calls base_fn(params), routing correctly.
            use_contender = params is not DEFAULT_WEIGHTS
            self._backends[tag] = ValueFunctionPlayer(
                self.color,
                value_fn_builder_name="C" if use_contender else None,
                params=params if use_contender else None,
                epsilon=epsilon,
            )
        return self._backends[tag]

    def reset_state(self):
        tag, params, epsilon = self._rng.choice(VFP_VARIANTS)
        self._active = self._get_backend(tag, params, epsilon)
        self._active_tag = tag
        if hasattr(self._active, "reset_state"):
            self._active.reset_state()

    def decide(self, game: Game, playable_actions: Iterable[Action]):
        if self._active is None:
            self.reset_state()
        return self._active.decide(game, playable_actions)
