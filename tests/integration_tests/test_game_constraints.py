import json

import pytest

from catanatron_server import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    fake_model_path = tmp_path / "fake_model.zip"
    fake_model_path.write_bytes(b"not-a-real-model")
    config_path = tmp_path / "models_config.json"
    config_path.write_text(
        json.dumps(
            [
                {"id": "HUMAN", "kind": "human", "label": "Human", "category": "human"},
                {
                    "id": "RANDOM",
                    "kind": "heuristic",
                    "label": "Random",
                    "category": "baseline",
                },
                {
                    "id": "CATANATRON",
                    "kind": "alphabeta",
                    "label": "AB",
                    "category": "baseline",
                },
                {
                    "id": "PPO",
                    "kind": "ppo",
                    "label": "PPO",
                    "category": "behavioral_cloning",
                    "model_path": str(fake_model_path),
                },
            ]
        )
    )
    monkeypatch.setenv("MODELS_CONFIG_PATH", str(config_path))
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    return app.test_client()


def test_model_players_rejected_for_non_1v1_games(client):
    response = client.post(
        "/api/games",
        data=json.dumps({"players": ["HUMAN", "PPO", "RANDOM"]}),
        content_type="application/json",
    )
    assert response.status_code == 400
    assert b"only in 1v1" in response.data


def test_non_model_multiplayer_games_still_allowed(client):
    response = client.post(
        "/api/games",
        data=json.dumps({"players": ["HUMAN", "RANDOM", "CATANATRON"]}),
        content_type="application/json",
    )
    assert response.status_code == 200
    assert "game_id" in response.get_json()
