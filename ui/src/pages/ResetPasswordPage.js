import React, { useMemo, useState } from "react";
import { useHistory, useLocation } from "react-router-dom";
import { Button, TextField, Typography } from "@material-ui/core";

import { resetPassword } from "../utils/apiClient";
import "./AuthPages.scss";

export default function ResetPasswordPage() {
  const history = useHistory();
  const location = useLocation();
  const token = useMemo(
    () => new URLSearchParams(location.search).get("token") || "",
    [location.search]
  );
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setMessage("");
    try {
      const payload = await resetPassword(token, password);
      setMessage(payload.message || "Password has been reset.");
    } catch (err) {
      setError(
        err.response?.data?.description ||
          "Could not reset password. The link may have expired."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="auth-page">
      <form className="auth-card" onSubmit={handleSubmit}>
        <Typography className="auth-title" variant="h5">
          Set New Password
        </Typography>
        <Typography className="auth-subtitle" variant="body2">
          Choose a new password for your account.
        </Typography>

        {!token && (
          <Typography className="auth-error" variant="body2">
            Missing reset token. Please request a new reset email.
          </Typography>
        )}

        <TextField
          className="auth-textfield"
          label="New Password"
          variant="outlined"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />

        {error && (
          <Typography className="auth-error" variant="body2">
            {error}
          </Typography>
        )}
        {message && (
          <Typography className="auth-success" variant="body2">
            {message}
          </Typography>
        )}

        <Button
          type="submit"
          variant="contained"
          color="primary"
          disabled={loading || !token}
        >
          {loading ? "Saving..." : "Reset Password"}
        </Button>
        <Button variant="outlined" onClick={() => history.push("/login")}>
          Go to Sign In
        </Button>
      </form>
    </main>
  );
}
