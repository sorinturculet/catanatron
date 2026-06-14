import json
import threading

from flask import current_app, request
from flask_socketio import SocketIO, join_room

from catanatron.json import GameEncoder
from catanatron_server.models import get_game_state, upsert_game_state

socketio = SocketIO(cors_allowed_origins="*", async_mode="threading")

_ACTIVE_GAMES_LOCK = threading.Lock()
_ACTIVE_BOT_GAMES = set()


def _room(game_id):
    return f"game:{game_id}"


def emit_state(game):
    emit_state_to_target(game, _room(game.id))


def emit_state_to_target(game, target):
    payload = json.loads(json.dumps(game, cls=GameEncoder))
    socketio.emit("state_updated", payload, to=target)
    winner = game.winning_color()
    if winner is not None:
        socketio.emit(
            "game_completed",
            {"game_id": game.id, "winner": winner.value},
            to=target,
        )


@socketio.on("subscribe")
def handle_subscribe(data):
    game_id = (data or {}).get("game_id")
    if not game_id:
        return
    join_room(_room(game_id))
    try:
        game = get_game_state(game_id)
    except Exception:
        return
    emit_state_to_target(game, request.sid)
    start_bot_turns_if_needed(current_app._get_current_object(), game_id)


def start_bot_turns_if_needed(app, game_id):
    with _ACTIVE_GAMES_LOCK:
        if game_id in _ACTIVE_BOT_GAMES:
            return
        _ACTIVE_BOT_GAMES.add(game_id)
    socketio.start_background_task(run_bot_turns, app, game_id)


def run_bot_turns(app, game_id):
    try:
        with app.app_context():
            delay = float(current_app.config.get("SOCKET_BOT_TICK_DELAY_SEC", 0.2))
            while True:
                game = get_game_state(game_id)
                if game.winning_color() is not None:
                    emit_state(game)
                    break
                if not game.state.current_player().is_bot:
                    break

                game.play_tick()
                upsert_game_state(game)
                emit_state(game)
                socketio.sleep(delay)
    finally:
        with _ACTIVE_GAMES_LOCK:
            _ACTIVE_BOT_GAMES.discard(game_id)
