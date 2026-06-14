import React, { useState } from "react";
import { useHistory } from "react-router-dom";
import { Button, TextField, Typography } from "@material-ui/core";

import { forgotPassword } from "../utils/apiClient";
import "./AuthPages.scss";

export default function ForgotPasswordPage() {
  const history = useHistory();
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setMessage("");
    try {
      const payload = await forgotPassword(email);
      setMessage(
        payload.message ||
          "If that account exists, a password reset email has been sent."
      );
    } catch (err) {
      setError(
        err.response?.data?.description ||
          "Could not start password reset. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="auth-page">
      <form className="auth-card" onSubmit={handleSubmit}>
        <Typography className="auth-title" variant="h5">
          Reset Password
        </Typography>
        <Typography className="auth-subtitle" variant="body2">
          Enter your email and we will send a reset link.
        </Typography>

        <TextField
          className="auth-textfield"
          label="Email"
          variant="outlined"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
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

        <Button type="submit" variant="contained" color="primary" disabled={loading}>
          {loading ? "Sending..." : "Send Reset Link"}
        </Button>
        <Button variant="outlined" onClick={() => history.push("/login")}>
          Back to Sign In
        </Button>
      </form>
    </main>
  );
}
