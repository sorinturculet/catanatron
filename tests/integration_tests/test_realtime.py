import json
import time

import pytest

from catanatron_server import create_app
from catanatron_server.models import get_game_state
from catanatron_server.realtime import emit_state, socketio


def _wait_for_event(socket_client, event_name, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for event in socket_client.get_received():
            if event["name"] == event_name:
                return event
        time.sleep(0.02)
    return None


@pytest.fixture
def app(tmp_path):
    db_path = tmp_path / "realtime_test.db"
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{db_path}",
            "SOCKET_BOT_TICK_DELAY_SEC": 0,
        }
    )
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_realtime_emits_state_updates_for_subscribed_game(client, app):
    create_response = client.post(
        "/api/games",
        data=json.dumps({"players": ["RANDOM", "RANDOM"]}),
        content_type="application/json",
    )
    game_id = create_response.get_json()["game_id"]

    socket_client = socketio.test_client(app, flask_test_client=client)
    socket_client.emit("subscribe", {"game_id": game_id})

    deadline = time.time() + 5
    events = []
    while time.time() < deadline:
        events.extend(socket_client.get_received())
        has_state = any(event["name"] == "state_updated" for event in events)
        if has_state:
            break
        time.sleep(0.05)

    socket_client.disconnect()

    state_events = [event for event in events if event["name"] == "state_updated"]
    assert len(state_events) > 0


def test_emit_state_emits_game_completed_event_when_winner(client, app):
    create_response = client.post(
        "/api/games",
        data=json.dumps({"players": ["RANDOM", "RANDOM"]}),
        content_type="application/json",
    )
    game_id = create_response.get_json()["game_id"]

    socket_client = socketio.test_client(app, flask_test_client=client)
    socket_client.emit("subscribe", {"game_id": game_id})

    with app.app_context():
        game = get_game_state(game_id)
        winner = game.state.colors[0]
        game.winning_color = lambda: winner
        emit_state(game)

    events = socket_client.get_received()
    socket_client.disconnect()
    completed_events = [event for event in events if event["name"] == "game_completed"]
    assert len(completed_events) > 0


def test_realtime_broadcasts_same_state_to_multiple_subscribers(client, app):
    create_response = client.post(
        "/api/games",
        data=json.dumps({"players": ["HUMAN", "HUMAN"]}),
        content_type="application/json",
    )
    game_id = create_response.get_json()["game_id"]

    first_client = socketio.test_client(app, flask_test_client=client)
    second_client = socketio.test_client(app, flask_test_client=client)
    first_client.emit("subscribe", {"game_id": game_id})
    second_client.emit("subscribe", {"game_id": game_id})

    # Drain initial per-subscriber state sync sent on subscribe.
    first_client.get_received()
    second_client.get_received()

    with app.app_context():
        game = get_game_state(game_id)
        emit_state(game)

    first_event = _wait_for_event(first_client, "state_updated")
    second_event = _wait_for_event(second_client, "state_updated")

    first_client.disconnect()
    second_client.disconnect()

    assert first_event is not None
    assert second_event is not None
    assert first_event["args"][0] == second_event["args"][0]
