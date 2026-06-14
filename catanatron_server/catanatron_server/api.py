import json
import logging
import traceback

from flask import Response, Blueprint, jsonify, abort, request, current_app
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request

from catanatron_server.models import GameOwnership, upsert_game_state, get_game_state, db
from catanatron_server.realtime import emit_state, start_bot_turns_if_needed
from catanatron.json import GameEncoder, action_from_json
from catanatron.models.player import Color
from catanatron.game import Game
from catanatron_experimental.machine_learning.players.minimax import AlphaBetaPlayer
from catanatron_experimental.analysis.mcts_analysis import GameAnalyzer

bp = Blueprint("api", __name__, url_prefix="/api")


def _extract_player_keys():
    payload = request.get_json(silent=True) or {}
    player_keys = payload.get("players")
    if not isinstance(player_keys, list) or len(player_keys) < 2 or len(player_keys) > 4:
        abort(400, description="`players` must be a list with 2 to 4 player ids.")
    return player_keys


def _assert_supported_player_constraints(player_keys):
    try:
        contains_model_player = any(
            current_app.registry.is_model_based_player(player_id) for player_id in player_keys
        )
    except ValueError as exc:
        abort(400, description=str(exc))

    if len(player_keys) != 2 and contains_model_player:
        abort(400, description="Model-based AI players are supported only in 1v1 games.")


@bp.route("/players", methods=("GET",))
def list_players_endpoint():
    return jsonify(current_app.registry.list_public())


@bp.route("/games", methods=("POST",))
def post_game_endpoint():
    player_keys = _extract_player_keys()
    _assert_supported_player_constraints(player_keys)
    try:
        players = [
            current_app.registry.create_player(player_key, color)
            for player_key, color in zip(player_keys, Color)
        ]
    except ValueError as exc:
        abort(400, description=str(exc))

    game = Game(players=players)
    upsert_game_state(game)
    emit_state(game)
    start_bot_turns_if_needed(current_app._get_current_object(), game.id)

    verify_jwt_in_request(optional=True)
    user_id = get_jwt_identity()
    if user_id is not None:
        db.session.add(
            GameOwnership(
                user_id=int(user_id),
                game_uuid=game.id,
                num_players=len(players),
                players_config=json.dumps(player_keys),
            )
        )
        db.session.commit()

    return jsonify({"game_id": game.id})


@bp.route("/games/<string:game_id>/states/<string:state_index>", methods=("GET",))
def get_game_endpoint(game_id, state_index):
    state_index = None if state_index == "latest" else int(state_index)
    game = get_game_state(game_id, state_index)
    if game is None:
        abort(404, description="Resource not found")

    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


@bp.route("/games/<string:game_id>/actions", methods=["POST"])
def post_action_endpoint(game_id):
    game = get_game_state(game_id)
    if game is None:
        abort(404, description="Resource not found")

    if game.winning_color() is not None:
        return Response(
            response=json.dumps(game, cls=GameEncoder),
            status=200,
            mimetype="application/json",
        )

    # TODO: remove `or body_is_empty` when fully implement actions in FE
    body_is_empty = (not request.data) or request.json is None
    if game.state.current_player().is_bot or body_is_empty:
        game.play_tick()
        upsert_game_state(game)
        emit_state(game)
    else:
        action = action_from_json(request.json)
        game.execute(action)
        upsert_game_state(game)
        emit_state(game)

    start_bot_turns_if_needed(current_app._get_current_object(), game.id)

    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


@bp.route("/stress-test", methods=["GET"])
def stress_test_endpoint():
    players = [
        AlphaBetaPlayer(Color.RED, 2, True),
        AlphaBetaPlayer(Color.BLUE, 2, True),
        AlphaBetaPlayer(Color.ORANGE, 2, True),
        AlphaBetaPlayer(Color.WHITE, 2, True),
    ]
    game = Game(players=players)
    game.play_tick()
    return Response(
        response=json.dumps(game, cls=GameEncoder),
        status=200,
        mimetype="application/json",
    )


@bp.route(
    "/games/<string:game_id>/states/<string:state_index>/mcts-analysis", methods=["GET"]
)
def mcts_analysis_endpoint(game_id, state_index):
    """Get MCTS analysis for specific game state."""
    logging.info(f"MCTS analysis request for game {game_id} at state {state_index}")

    try:
        # Convert 'latest' to None for consistency with get_game_state
        state_index = None if state_index == "latest" else int(state_index)

        game = get_game_state(game_id, state_index)
        if game is None:
            logging.error(f"Game/state not found: {game_id}/{state_index}")
            abort(404, description="Game state not found")

        analyzer = GameAnalyzer(num_simulations=100)
        probabilities = analyzer.analyze_win_probabilities(game)

        logging.info(f"Analysis successful. Probabilities: {probabilities}")
        return Response(
            response=json.dumps(
                {
                    "success": True,
                    "probabilities": probabilities,
                    "state_index": state_index
                    if state_index is not None
                    else len(game.state.actions),
                }
            ),
            status=200,
            mimetype="application/json",
        )

    except Exception as e:
        logging.error(f"Error in MCTS analysis endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return Response(
            response=json.dumps(
                {"success": False, "error": str(e), "trace": traceback.format_exc()}
            ),
            status=500,
            mimetype="application/json",
        )


# ===== Debugging Routes
# @app.route(
#     "/games/<string:game_id>/players/<int:player_index>/features", methods=["GET"]
# )
# def get_game_feature_vector(game_id, player_index):
#     game = get_game_state(game_id)
#     if game is None:
#         abort(404, description="Resource not found")

#     return create_sample(game, game.state.colors[player_index])


# @app.route("/games/<string:game_id>/value-function", methods=["GET"])
# def get_game_value_function(game_id):
#     game = get_game_state(game_id)
#     if game is None:
#         abort(404, description="Resource not found")

#     # model = tf.keras.models.load_model("data/models/mcts-rep-a")
#     model2 = tf.keras.models.load_model("data/models/mcts-rep-b")
#     feature_ordering = get_feature_ordering()
#     indices = [feature_ordering.index(f) for f in NUMERIC_FEATURES]
#     data = {}
#     for color in game.state.colors:
#         sample = create_sample_vector(game, color)
#         # scores = model.call(tf.convert_to_tensor([sample]))

#         inputs1 = [create_board_tensor(game, color)]
#         inputs2 = [[float(sample[i]) for i in indices]]
#         scores2 = model2.call(
#             [tf.convert_to_tensor(inputs1), tf.convert_to_tensor(inputs2)]
#         )
#         data[color.value] = float(scores2.numpy()[0][0])

#     return data
