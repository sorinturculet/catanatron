import os
import json
import pickle
from contextlib import contextmanager

from sqlalchemy import (
    MetaData,
    Column,
    Integer,
    String,
    LargeBinary,
    DateTime,
    ForeignKey,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from flask_sqlalchemy import SQLAlchemy

from catanatron.json import GameEncoder

# Using approach from: https://stackoverflow.com/questions/41004540/using-sqlalchemy-models-in-and-out-of-flask/41014157
metadata = MetaData()
Base = declarative_base(metadata=metadata)


class GameState(Base):
    __tablename__ = "game_states"

    id = Column(Integer, primary_key=True)
    uuid = Column(String(64), nullable=False)
    state_index = Column(Integer, nullable=False)
    state = Column(String, nullable=False)
    pickle_data = Column(LargeBinary, nullable=False)

    # TODO: unique uuid and state_index

    @staticmethod
    def from_game(game):
        state = json.dumps(game, cls=GameEncoder)
        pickle_data = pickle.dumps(game, pickle.HIGHEST_PROTOCOL)
        return GameState(
            uuid=game.id,
            state_index=len(game.state.actions),
            state=state,
            pickle_data=pickle_data,
        )


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(254), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())


class GameOwnership(Base):
    __tablename__ = "game_ownerships"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    game_uuid = Column(String(64), nullable=False, index=True)
    num_players = Column(Integer, nullable=False)
    players_config = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("user_id", "game_uuid", name="uq_game_ownership_user_game"),
    )


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(64), nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    expires_at = Column(DateTime, nullable=False, index=True)
    used_at = Column(DateTime, nullable=True, index=True)


db = SQLAlchemy(metadata=metadata)


@contextmanager
def database_session():
    """Can use like:
    with database_session() as session:
        game_states = session.query(GameState).all()
    """
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://catanatron:victorypoint@127.0.0.1:5432/catanatron_db",
    )
    engine = create_engine(database_url)
    session = Session(engine)
    try:
        yield session
    finally:
        session.expunge_all()
        session.close()


def upsert_game_state(game, session_param=None):
    game_state = GameState.from_game(game)
    session = session_param or db.session
    session.add(game_state)
    session.commit()
    return game_state


def get_game_state(game_id, state_index=None):
    if state_index is None:
        result = (
            db.session.query(GameState)
            .filter_by(uuid=game_id)
            .order_by(GameState.state_index.desc())
            .first_or_404()
        )
    else:
        result = (
            db.session.query(GameState)
            .filter_by(uuid=game_id, state_index=state_index)
            .first_or_404()
        )
    db.session.commit()
    game = pickle.loads(result.pickle_data)
    return game
