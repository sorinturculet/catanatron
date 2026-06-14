import json
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

from catanatron.models.player import Color, RandomPlayer
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron_experimental.machine_learning.players.ppo import PPOPlayer
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer

LOGGER = logging.getLogger(__name__)

VALID_KINDS = {"human", "heuristic", "alphabeta", "ppo"}
VALID_CATEGORIES = {"baseline", "expert_guided", "behavioral_cloning", "human"}


def _load_sb3_model(model_path: str):
    from sb3_contrib import MaskablePPO

    from catanatron_experimental.machine_learning.custom_cnn import CustomCNN

    custom_objects = {"features_extractor_class": CustomCNN}

    try:
        return MaskablePPO.load(model_path, custom_objects=custom_objects)
    except Exception:
        from catanatron_experimental.machine_learning.moe_policy import MoEMaskablePolicy

        custom_objects["policy_class"] = MoEMaskablePolicy
        return MaskablePPO.load(model_path, custom_objects=custom_objects)


class PlayerDescriptor:
    def __init__(
        self,
        player_id: str,
        label: str,
        kind: str,
        description: str = "",
        model_path: Optional[str] = None,
        category: str = "baseline",
    ):
        self.id = player_id
        self.label = label
        self.description = description
        self.kind = kind
        self.model_path = model_path
        self.category = category

    def to_public_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "kind": self.kind,
            "category": self.category,
        }


class ModelRegistry:
    def __init__(
        self,
        entries: List[PlayerDescriptor],
        load_ppo_fn: Optional[Callable[[str], object]] = None,
    ):
        if not entries:
            raise ValueError("Model registry must include at least one entry.")

        self._entries = entries
        self._entries_by_id = {}
        for entry in entries:
            if entry.id in self._entries_by_id:
                raise ValueError("Duplicate player id in model registry: {}".format(entry.id))
            self._entries_by_id[entry.id] = entry

        max_loaded_models = int(os.environ.get("MAX_LOADED_MODELS", "2"))
        max_loaded_models = max(1, max_loaded_models)
        self._load_ppo_model = lru_cache(maxsize=max_loaded_models)(
            load_ppo_fn or _load_sb3_model
        )
        self._load_lock = threading.Lock()

    @classmethod
    def from_json(cls, path: Path) -> "ModelRegistry":
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise ValueError("Model config not found: {}".format(config_path))

        payload = json.loads(config_path.read_text())
        if not isinstance(payload, list):
            raise ValueError("Model config must be a JSON array.")

        repo_root = Path(__file__).resolve().parents[2]
        entries = []

        for idx, item in enumerate(payload):
            entry = cls._parse_entry(item, idx, repo_root)
            if entry is not None:
                entries.append(entry)

        if not entries:
            raise ValueError("Model config resulted in zero usable entries.")

        return cls(entries)

    @staticmethod
    def _parse_entry(item: object, index: int, repo_root: Path) -> Optional[PlayerDescriptor]:
        if not isinstance(item, dict):
            raise ValueError("Entry #{} must be a JSON object.".format(index))

        allowed_fields = {"id", "label", "description", "kind", "model_path", "category"}
        extra_fields = set(item.keys()) - allowed_fields
        if extra_fields:
            raise ValueError(
                "Entry #{} has unknown fields: {}".format(index, sorted(extra_fields))
            )

        for required in ("id", "label", "kind"):
            if required not in item:
                raise ValueError("Entry #{} missing required field '{}'.".format(index, required))

        player_id = item["id"]
        label = item["label"]
        description = item.get("description", "")
        kind = item["kind"]
        category = item.get("category", "baseline")
        model_path = item.get("model_path")

        if not isinstance(player_id, str) or not player_id:
            raise ValueError("Entry #{} has invalid 'id'.".format(index))
        if not isinstance(label, str) or not label:
            raise ValueError("Entry #{} has invalid 'label'.".format(index))
        if not isinstance(description, str):
            raise ValueError("Entry #{} has invalid 'description'.".format(index))
        if kind not in VALID_KINDS:
            raise ValueError("Entry #{} has invalid kind '{}'.".format(index, kind))
        if category not in VALID_CATEGORIES:
            raise ValueError("Entry #{} has invalid category '{}'.".format(index, category))

        resolved_model_path = None
        if kind == "ppo":
            if not isinstance(model_path, str) or not model_path:
                raise ValueError("PPO entry '{}' must provide a string model_path.".format(player_id))
            model_path_obj = Path(model_path)
            if not model_path_obj.is_absolute():
                model_path_obj = (repo_root / model_path_obj).resolve()
            if not model_path_obj.exists():
                LOGGER.warning(
                    "Skipping registry entry '%s': model file not found at %s",
                    player_id,
                    model_path_obj,
                )
                return None
            resolved_model_path = str(model_path_obj)
        elif model_path is not None:
            raise ValueError(
                "Only PPO entries may define model_path (entry '{}').".format(player_id)
            )

        return PlayerDescriptor(
            player_id=player_id,
            label=label,
            description=description,
            kind=kind,
            model_path=resolved_model_path,
            category=category,
        )

    def list_public(self) -> List[Dict[str, str]]:
        return [entry.to_public_dict() for entry in self._entries]

    def create_player(self, player_id: str, color: Color):
        descriptor = self._entries_by_id.get(player_id)
        if descriptor is None:
            raise ValueError("Unknown player id: {}".format(player_id))

        if descriptor.kind == "human":
            return ValueFunctionPlayer(color, is_bot=False)
        if descriptor.kind == "heuristic":
            return self._create_heuristic_player(descriptor, color)
        if descriptor.kind == "alphabeta":
            return AlphaBetaPlayer(color, 2, True)
        if descriptor.kind == "ppo":
            return self._create_ppo_player(descriptor, color)

        raise ValueError("Unsupported player kind '{}'.".format(descriptor.kind))

    def is_model_based_player(self, player_id: str) -> bool:
        descriptor = self._entries_by_id.get(player_id)
        if descriptor is None:
            raise ValueError("Unknown player id: {}".format(player_id))
        return descriptor.kind == "ppo"

    @staticmethod
    def _create_heuristic_player(descriptor: PlayerDescriptor, color: Color):
        if descriptor.id == "RANDOM":
            return RandomPlayer(color)
        if descriptor.id == "VALUE_FUNCTION":
            return ValueFunctionPlayer(color)
        raise ValueError(
            "Unsupported heuristic player id '{}'. Add explicit mapping in model_registry.py.".format(
                descriptor.id
            )
        )

    def _create_ppo_player(self, descriptor: PlayerDescriptor, color: Color):
        player = PPOPlayer(color, model_path="")
        player.model_path = descriptor.model_path
        with self._load_lock:
            player.model = self._load_ppo_model(descriptor.model_path)
        return player
