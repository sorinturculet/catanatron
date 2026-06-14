import React, { useEffect, useState } from "react";
import { useHistory } from "react-router-dom";
import { Button, TextField, Typography } from "@material-ui/core";

import { useAuth } from "../auth";
import "./AuthPages.scss";

export default function RegisterPage() {
  const history = useHistory();
  const { register, isAuthenticated } = useAuth();
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
      await register(email, password);
      history.push("/me");
    } catch (err) {
      setError(
        err.response?.data?.description || "Registration failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="auth-page">
      <form className="auth-card" onSubmit={handleSubmit}>
        <Typography className="auth-title" variant="h5">
          Register
        </Typography>
        <Typography className="auth-subtitle" variant="body2">
          Create an account to save your games.
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
          {loading ? "Creating account..." : "Create Account"}
        </Button>

        <Typography className="auth-footer" variant="caption">
          Already have an account?
          <span className="auth-link" onClick={() => history.push("/login")}>
            Sign in
          </span>
        </Typography>
      </form>
    </main>
  );
}
