import json

import pytest

from catanatron_server import create_app


@pytest.fixture
def app():
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def register_and_get_token(client, email):
    response = client.post(
        "/api/auth/register",
        data=json.dumps({"email": email, "password": "hunter22"}),
        content_type="application/json",
    )
    return response.get_json()["access_token"]


def create_game(client, players, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return client.post(
        "/api/games",
        data=json.dumps({"players": players}),
        headers=headers,
    )


def test_my_games_only_returns_authenticated_users_games(client):
    token_a = register_and_get_token(client, "a@example.com")
    token_b = register_and_get_token(client, "b@example.com")

    create_game(client, ["HUMAN", "RANDOM"], token=token_a)
    create_game(client, ["HUMAN", "CATANATRON"], token=token_b)
    create_game(client, ["HUMAN", "RANDOM"], token=token_a)

    response_a = client.get(
        "/api/me/games", headers={"Authorization": f"Bearer {token_a}"}
    )
    response_b = client.get(
        "/api/me/games", headers={"Authorization": f"Bearer {token_b}"}
    )

    payload_a = response_a.get_json()
    payload_b = response_b.get_json()
    assert response_a.status_code == 200
    assert response_b.status_code == 200
    assert payload_a["total"] == 2
    assert payload_b["total"] == 1


def test_anonymous_game_creation_still_works(client):
    response = create_game(client, ["HUMAN", "RANDOM"])
    assert response.status_code == 200
    assert "game_id" in response.get_json()


def test_my_games_requires_auth(client):
    response = client.get("/api/me/games")
    assert response.status_code == 401
