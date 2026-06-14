import json
from datetime import datetime, timedelta

import pytest

from catanatron_server import create_app
from catanatron_server.models import PasswordResetToken, db


@pytest.fixture
def app():
    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"})
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_register_and_me_flow(client):
    register_response = client.post(
        "/api/auth/register",
        data=json.dumps({"email": "test@example.com", "password": "hunter22"}),
        content_type="application/json",
    )
    payload = register_response.get_json()

    assert register_response.status_code == 201
    assert "access_token" in payload
    assert payload["user"]["email"] == "test@example.com"

    me_response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {payload['access_token']}"},
    )
    me_payload = me_response.get_json()
    assert me_response.status_code == 200
    assert me_payload["email"] == "test@example.com"


def test_register_duplicate_email_returns_409(client):
    body = json.dumps({"email": "dupe@example.com", "password": "hunter22"})
    client.post("/api/auth/register", data=body, content_type="application/json")
    response = client.post("/api/auth/register", data=body, content_type="application/json")
    assert response.status_code == 409


def test_login_success_and_wrong_password(client):
    client.post(
        "/api/auth/register",
        data=json.dumps({"email": "login@example.com", "password": "hunter22"}),
        content_type="application/json",
    )

    ok = client.post(
        "/api/auth/login",
        data=json.dumps({"email": "login@example.com", "password": "hunter22"}),
        content_type="application/json",
    )
    assert ok.status_code == 200
    assert "access_token" in ok.get_json()

    bad = client.post(
        "/api/auth/login",
        data=json.dumps({"email": "login@example.com", "password": "wrongpass"}),
        content_type="application/json",
    )
    assert bad.status_code == 401


def test_me_requires_auth(client):
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_forgot_password_generic_message_for_unknown_email(client):
    response = client.post(
        "/api/auth/forgot-password",
        data=json.dumps({"email": "missing@example.com"}),
        content_type="application/json",
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert "message" in payload


def test_reset_password_flow(client):
    client.post(
        "/api/auth/register",
        data=json.dumps({"email": "reset@example.com", "password": "oldpass1"}),
        content_type="application/json",
    )

    forgot = client.post(
        "/api/auth/forgot-password",
        data=json.dumps({"email": "reset@example.com"}),
        content_type="application/json",
    )
    forgot_payload = forgot.get_json()
    assert forgot.status_code == 200
    assert "reset_token" in forgot_payload

    reset = client.post(
        "/api/auth/reset-password",
        data=json.dumps(
            {"token": forgot_payload["reset_token"], "password": "newpass2"}
        ),
        content_type="application/json",
    )
    assert reset.status_code == 200

    bad_old = client.post(
        "/api/auth/login",
        data=json.dumps({"email": "reset@example.com", "password": "oldpass1"}),
        content_type="application/json",
    )
    assert bad_old.status_code == 401

    ok_new = client.post(
        "/api/auth/login",
        data=json.dumps({"email": "reset@example.com", "password": "newpass2"}),
        content_type="application/json",
    )
    assert ok_new.status_code == 200


def test_reset_password_token_cannot_be_reused(client):
    client.post(
        "/api/auth/register",
        data=json.dumps({"email": "reuse@example.com", "password": "start123"}),
        content_type="application/json",
    )
    forgot = client.post(
        "/api/auth/forgot-password",
        data=json.dumps({"email": "reuse@example.com"}),
        content_type="application/json",
    )
    token = forgot.get_json()["reset_token"]

    first = client.post(
        "/api/auth/reset-password",
        data=json.dumps({"token": token, "password": "next123"}),
        content_type="application/json",
    )
    assert first.status_code == 200

    second = client.post(
        "/api/auth/reset-password",
        data=json.dumps({"token": token, "password": "third123"}),
        content_type="application/json",
    )
    assert second.status_code == 400


def test_reset_password_rejects_expired_token(client, app):
    client.post(
        "/api/auth/register",
        data=json.dumps({"email": "expired@example.com", "password": "start123"}),
        content_type="application/json",
    )
    forgot = client.post(
        "/api/auth/forgot-password",
        data=json.dumps({"email": "expired@example.com"}),
        content_type="application/json",
    )
    token = forgot.get_json()["reset_token"]

    with app.app_context():
        row = db.session.query(PasswordResetToken).first()
        row.expires_at = datetime.utcnow() - timedelta(minutes=1)
        db.session.commit()

    response = client.post(
        "/api/auth/reset-password",
        data=json.dumps({"token": token, "password": "next123"}),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_reset_password_validates_input(client):
    missing_token = client.post(
        "/api/auth/reset-password",
        data=json.dumps({"password": "next123"}),
        content_type="application/json",
    )
    assert missing_token.status_code == 400

    short_password = client.post(
        "/api/auth/reset-password",
        data=json.dumps({"token": "abc", "password": "123"}),
        content_type="application/json",
    )
    assert short_password.status_code == 400
