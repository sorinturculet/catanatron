import React, { useEffect, useState } from "react";
import { useHistory, useLocation } from "react-router-dom";
import { Button, TextField, Typography } from "@material-ui/core";

import { useAuth } from "../auth";
import "./AuthPages.scss";

export default function LoginPage() {
  const history = useHistory();
  const location = useLocation();
  const { login, isAuthenticated } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (isAuthenticated) {
      history.replace("/me");
    }
  }, [isAuthenticated, history]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      await login(email, password);
      const from = location.state?.from || "/me";
      history.push(from);
    } catch (err) {
      setError(err.response?.data?.description || "Login failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="auth-page">
      <form className="auth-card" onSubmit={handleSubmit}>
        <Typography className="auth-title" variant="h5">
          Sign In
        </Typography>
        <Typography className="auth-subtitle" variant="body2">
          Access your account and game history.
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
        <TextField
          className="auth-textfield"
          label="Password"
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

        <Button type="submit" variant="contained" color="primary" disabled={loading}>
          {loading ? "Signing in..." : "Sign In"}
        </Button>

        <Typography className="auth-footer" variant="caption">
          Need an account?
          <span className="auth-link" onClick={() => history.push("/register")}>
            Register
          </span>
        </Typography>
        <Typography className="auth-footer" variant="caption">
          Forgot password?
          <span className="auth-link" onClick={() => history.push("/forgot-password")}>
            Reset it
          </span>
        </Typography>
      </form>
    </main>
  );
}
