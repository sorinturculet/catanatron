import hashlib
import secrets
from datetime import datetime, timedelta

from flask import Blueprint, abort, current_app, jsonify, request
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required
from werkzeug.security import check_password_hash, generate_password_hash

from catanatron_server.emailer import send_password_reset_email
from catanatron_server.models import PasswordResetToken, User, db

bp = Blueprint("auth", __name__, url_prefix="/api/auth")


def _read_credentials():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    return email, password


def _serialize_user(user):
    return {
        "id": user.id,
        "email": user.email,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _hash_reset_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _issue_password_reset_token(user_id):
    token = secrets.token_urlsafe(48)
    ttl_minutes = int(current_app.config.get("PASSWORD_RESET_TOKEN_TTL_MINUTES", 30))
    expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)
    db.session.add(
        PasswordResetToken(
            user_id=user_id,
            token_hash=_hash_reset_token(token),
            expires_at=expires_at,
        )
    )
    db.session.commit()
    return token


@bp.route("/register", methods=["POST"])
def register_endpoint():
    email, password = _read_credentials()
    if not email or "@" not in email:
        abort(400, description="Please provide a valid email.")
    if len(password) < 6:
        abort(400, description="Password must be at least 6 characters.")

    existing = db.session.query(User).filter_by(email=email).first()
    if existing is not None:
        abort(409, description="Email already registered.")

    user = User(
        email=email,
        password_hash=generate_password_hash(password, method="pbkdf2:sha256"),
    )
    db.session.add(user)
    db.session.commit()

    token = create_access_token(identity=str(user.id))
    return jsonify(access_token=token, user=_serialize_user(user)), 201


@bp.route("/login", methods=["POST"])
def login_endpoint():
    email, password = _read_credentials()
    user = db.session.query(User).filter_by(email=email).first()
    if user is None or not check_password_hash(user.password_hash, password):
        abort(401, description="Invalid email or password.")

    token = create_access_token(identity=str(user.id))
    return jsonify(access_token=token, user=_serialize_user(user))


@bp.route("/me", methods=["GET"])
@jwt_required()
def me_endpoint():
    user_id = int(get_jwt_identity())
    user = db.session.query(User).filter_by(id=user_id).first()
    if user is None:
        abort(404, description="User not found.")
    return jsonify(_serialize_user(user))


@bp.route("/forgot-password", methods=["POST"])
def forgot_password_endpoint():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    if not email or "@" not in email:
        abort(400, description="Please provide a valid email.")

    response_payload = {
        "message": "If that account exists, a password reset email has been sent."
    }

    user = db.session.query(User).filter_by(email=email).first()
    if user is None:
        return jsonify(response_payload)

    token = _issue_password_reset_token(user.id)
    reset_url = current_app.config.get("PASSWORD_RESET_URL")
    reset_link = f"{reset_url}?token={token}"
    try:
        send_password_reset_email(user.email, reset_link)
    except Exception as exc:
        current_app.logger.exception("Failed to send password reset email: %s", exc)

    if current_app.config.get("TESTING"):
        response_payload["reset_token"] = token
    return jsonify(response_payload)


@bp.route("/reset-password", methods=["POST"])
def reset_password_endpoint():
    payload = request.get_json(silent=True) or {}
    token = (payload.get("token") or "").strip()
    password = payload.get("password") or ""
    if not token:
        abort(400, description="Reset token is required.")
    if len(password) < 6:
        abort(400, description="Password must be at least 6 characters.")

    token_hash = _hash_reset_token(token)
    now = datetime.utcnow()
    token_row = (
        db.session.query(PasswordResetToken)
        .filter(
            PasswordResetToken.token_hash == token_hash,
            PasswordResetToken.used_at.is_(None),
            PasswordResetToken.expires_at >= now,
        )
        .first()
    )
    if token_row is None:
        abort(400, description="Invalid or expired reset token.")

    user = db.session.query(User).filter_by(id=token_row.user_id).first()
    if user is None:
        abort(400, description="Invalid or expired reset token.")

    user.password_hash = generate_password_hash(password, method="pbkdf2:sha256")
    token_row.used_at = now

    # Invalidate any other outstanding reset tokens for this user.
    (
        db.session.query(PasswordResetToken)
        .filter(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used_at.is_(None),
            PasswordResetToken.id != token_row.id,
        )
        .update({"used_at": now}, synchronize_session=False)
    )
    db.session.commit()

    return jsonify({"message": "Password has been reset successfully."})
