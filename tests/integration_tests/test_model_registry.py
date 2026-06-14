import json
from pathlib import Path

import pytest

from catanatron.models.player import Color, RandomPlayer
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron_server import create_app
from catanatron_server.model_registry import ModelRegistry, PlayerDescriptor


@pytest.fixture
def app():
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_players_endpoint_lists_public_registry(client):
    response = client.get("/api/players")
    payload = response.get_json()

    assert response.status_code == 200
    assert response.is_json
    assert isinstance(payload, list)
    assert "HUMAN" in [entry["id"] for entry in payload]
    assert all("model_path" not in entry for entry in payload)


def test_registry_parses_shipped_config():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "catanatron_server"
        / "catanatron_server"
        / "models_config.json"
    )
    registry = ModelRegistry.from_json(config_path)

    player_ids = [entry["id"] for entry in registry.list_public()]
    assert "HUMAN" in player_ids
    assert "RANDOM" in player_ids
    assert "CATANATRON" in player_ids


def test_registry_raises_on_invalid_shape(tmp_path):
    config_path = tmp_path / "bad_models_config.json"
    config_path.write_text(json.dumps([{"id": "BROKEN", "label": "Broken"}]))

    with pytest.raises(ValueError):
        ModelRegistry.from_json(config_path)


def test_registry_creates_expected_player_subclasses(tmp_path):
    config_path = tmp_path / "models_config.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "id": "HUMAN",
                    "kind": "human",
                    "label": "You",
                    "category": "human",
                },
                {
                    "id": "RANDOM",
                    "kind": "heuristic",
                    "label": "Random",
                    "category": "baseline",
                },
                {
                    "id": "VALUE_FUNCTION",
                    "kind": "heuristic",
                    "label": "VFP",
                    "category": "baseline",
                },
                {
                    "id": "CATANATRON",
                    "kind": "alphabeta",
                    "label": "AB",
                    "category": "baseline",
                },
            ]
        )
    )

    registry = ModelRegistry.from_json(config_path)

    human_player = registry.create_player("HUMAN", Color.RED)
    assert isinstance(human_player, ValueFunctionPlayer)
    assert human_player.is_bot is False

    random_player = registry.create_player("RANDOM", Color.BLUE)
    assert isinstance(random_player, RandomPlayer)

    vfp_player = registry.create_player("VALUE_FUNCTION", Color.ORANGE)
    assert isinstance(vfp_player, ValueFunctionPlayer)
    assert vfp_player.is_bot is True

    alphabeta_player = registry.create_player("CATANATRON", Color.WHITE)
    assert isinstance(alphabeta_player, AlphaBetaPlayer)


def test_ppo_loader_is_cached_across_multiple_players():
    calls = []
    fake_model = object()

    def fake_loader(model_path):
        calls.append(model_path)
        return fake_model

    registry = ModelRegistry(
        entries=[
            PlayerDescriptor(
                player_id="PPO_TEST",
                label="PPO Test",
                description="",
                kind="ppo",
                model_path="/tmp/fake_model.zip",
                category="behavioral_cloning",
            )
        ],
        load_ppo_fn=fake_loader,
    )

    player_a = registry.create_player("PPO_TEST", Color.RED)
    player_b = registry.create_player("PPO_TEST", Color.BLUE)

    assert len(calls) == 1
    assert player_a.model is fake_model
    assert player_b.model is fake_model
