import json
from urllib import request as urlrequest

from flask import current_app

try:
    import resend
except ImportError:  # pragma: no cover - fallback path for local/debug
    resend = None


def send_password_reset_email(email: str, reset_link: str) -> bool:
    email_mode = current_app.config.get("EMAIL_MODE", "log")
    if email_mode == "log":
        current_app.logger.info("Password reset link for %s: %s", email, reset_link)
        return True
    if email_mode != "resend":
        current_app.logger.warning("Unknown EMAIL_MODE=%s; falling back to log.", email_mode)
        current_app.logger.info("Password reset link for %s: %s", email, reset_link)
        return True

    api_key = current_app.config.get("RESEND_API_KEY")
    from_email = current_app.config.get("RESEND_FROM_EMAIL", "onboarding@resend.dev")
    if not api_key:
        current_app.logger.warning("RESEND_API_KEY is not configured.")
        return False

    payload = {
        "from": from_email,
        "to": [email],
        "subject": "Reset your Catanatron password",
        "html": (
            "<p>We received a request to reset your password.</p>"
            f"<p><a href=\"{reset_link}\">Reset password</a></p>"
            "<p>If you did not request this, you can ignore this email.</p>"
        ),
    }

    # Prefer SDK-style sending (similar to the JS Resend example).
    if resend is not None:
        resend.api_key = api_key
        response = resend.Emails.send(payload)
        return bool(response)

    # Fallback to direct HTTP API if SDK is unavailable.
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        "https://api.resend.com/emails",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urlrequest.urlopen(req, timeout=10) as response:
        return response.status < 400
