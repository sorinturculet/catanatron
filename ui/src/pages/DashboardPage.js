import React, { useEffect, useState } from "react";
import { useHistory } from "react-router-dom";
import {
  Button,
  CircularProgress,
  Divider,
  Paper,
  Typography,
} from "@material-ui/core";

import { useAuth } from "../auth";
import { getMyGames } from "../utils/apiClient";
import "./DashboardPage.scss";

function describePlayers(players) {
  if (!Array.isArray(players) || players.length === 0) return "Unknown setup";
  return players.join(" vs ");
}

function shortGameId(gameId) {
  if (!gameId) return "Unknown";
  return gameId.slice(0, 8).toUpperCase();
}

function formatCreatedAt(value) {
  if (!value) return "Unknown date";
  return new Date(value).toLocaleString();
}

export default function DashboardPage() {
  const history = useHistory();
  const { user, logout } = useAuth();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [games, setGames] = useState([]);

  useEffect(() => {
    (async () => {
      try {
        const payload = await getMyGames();
        setGames(payload.items || []);
      } catch (err) {
        setError(err.response?.data?.description || "Could not load game history.");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const openGame = (gameId) => history.push(`/games/${gameId}`);

  return (
    <main className="dashboard-page">
      <Paper className="dashboard-card">
        <div className="dashboard-header">
          <div>
            <Typography variant="h5" className="dashboard-title">
              My Games
            </Typography>
            <Typography variant="body2" className="dashboard-subtitle">
              {user?.email}
            </Typography>
          </div>
          <div className="dashboard-actions">
            <Button variant="outlined" size="small" onClick={() => history.push("/")}>
              Home
            </Button>
            <Button variant="outlined" size="small" onClick={logout}>
              Sign Out
            </Button>
          </div>
        </div>
        <Divider />

        {loading && (
          <div className="dashboard-loading">
            <CircularProgress size={28} />
          </div>
        )}

        {!loading && error && (
          <Typography className="dashboard-error" variant="body2">
            {error}
          </Typography>
        )}

        {!loading && !error && games.length === 0 && (
          <Typography className="dashboard-empty" variant="body2">
            No saved games yet. Start one from the Home page while logged in.
          </Typography>
        )}

        {!loading && !error && games.length > 0 && (
          <div className="dashboard-games-grid">
            {games.map((game) => (
              <Paper
                key={game.game_id}
                className="game-entry"
                variant="outlined"
                onClick={() => openGame(game.game_id)}
              >
                <div className="game-entry-main">
                  <Typography className="game-entry-title">
                    Game #{shortGameId(game.game_id)}
                  </Typography>
                  <Typography className="game-entry-meta" variant="caption">
                    {formatCreatedAt(game.created_at)}
                  </Typography>
                  <Typography className="game-entry-players" variant="body2">
                    {describePlayers(game.players_config)}
                  </Typography>
                </div>
                <div className="game-entry-actions">
                  <Button
                    variant="contained"
                    color="primary"
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      openGame(game.game_id);
                    }}
                  >
                    Open
                  </Button>
                </div>
              </Paper>
            ))}
          </div>
        )}
      </Paper>
    </main>
  );
}
