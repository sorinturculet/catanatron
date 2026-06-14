import os
from pathlib import Path
from datetime import timedelta

from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from catanatron_server.realtime import socketio

jwt = JWTManager()
migrate = Migrate()


def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    CORS(app)

    # ===== Load base configuration
    database_url = os.environ.get("DATABASE_URL", "sqlite:///:memory:")
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    secret_key = os.environ.get("SECRET_KEY", "dev")
    app.config.from_mapping(
        SECRET_KEY=secret_key,
        JWT_SECRET_KEY=os.environ.get("JWT_SECRET_KEY", secret_key),
        JWT_ACCESS_TOKEN_EXPIRES=timedelta(days=7),
        SQLALCHEMY_DATABASE_URI=database_url,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        EMAIL_MODE=os.environ.get("EMAIL_MODE", "log"),
        RESEND_API_KEY=os.environ.get("RESEND_API_KEY"),
        RESEND_FROM_EMAIL=os.environ.get("RESEND_FROM_EMAIL", "onboarding@resend.dev"),
        PASSWORD_RESET_TOKEN_TTL_MINUTES=int(
            os.environ.get("PASSWORD_RESET_TOKEN_TTL_MINUTES", "30")
        ),
        PASSWORD_RESET_URL=os.environ.get(
            "PASSWORD_RESET_URL", "http://localhost:3000/reset-password"
        ),
        SOCKET_BOT_TICK_DELAY_SEC=float(os.environ.get("SOCKET_BOT_TICK_DELAY_SEC", "0.2")),
    )
    if test_config is not None:
        app.config.update(test_config)

    # ===== Initialize Database
    from catanatron_server.models import db

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)
    socketio.init_app(app)

    should_bootstrap_tables = app.config.get("TESTING") or os.environ.get(
        "AUTO_CREATE_TABLES", "0"
    ) == "1"
    with app.app_context():
        # Preferred path is `flask --app catanatron_server:create_app db upgrade`.
        # Keep create_all only for tests or explicit bootstrap mode.
        if should_bootstrap_tables:
            db.create_all()

    # ===== Initialize Model Registry
    from .model_registry import ModelRegistry

    config_path = Path(
        os.environ.get(
            "MODELS_CONFIG_PATH",
            str(Path(__file__).parent / "models_config.json"),
        )
    )
    app.registry = ModelRegistry.from_json(config_path)

    # ===== Initialize Routes
    from . import api, auth, history

    app.register_blueprint(api.bp)
    app.register_blueprint(auth.bp)
    app.register_blueprint(history.bp)

    return app
