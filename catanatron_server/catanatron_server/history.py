import json

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from catanatron_server.models import GameOwnership, db

bp = Blueprint("history", __name__, url_prefix="/api/me")


def _safe_json_loads(value):
    try:
        return json.loads(value)
    except Exception:
        return None


def _serialize_ownership(row):
    return {
        "game_id": row.game_uuid,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "num_players": row.num_players,
        "players_config": _safe_json_loads(row.players_config) or [],
    }


@bp.route("/games", methods=["GET"])
@jwt_required()
def my_games_endpoint():
    user_id = int(get_jwt_identity())
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(100, max(1, int(request.args.get("per_page", 20))))

    query = db.session.query(GameOwnership).filter_by(user_id=user_id)
    total = query.count()
    items = (
        query.order_by(GameOwnership.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    return jsonify(
        {
            "items": [_serialize_ownership(row) for row in items],
            "total": total,
            "page": page,
            "per_page": per_page,
        }
    )
