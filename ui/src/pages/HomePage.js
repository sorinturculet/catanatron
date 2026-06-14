import React, { useEffect, useMemo, useState } from "react";
import { useHistory } from "react-router-dom";
import { Button, Typography } from "@material-ui/core";
import Loader from "react-loader-spinner";
import { createGame, getPlayers } from "../utils/apiClient";
import { useAuth } from "../auth";

import "./HomePage.scss";

const CATEGORY_ORDER = [
  "human",
  "baseline",
  "expert_guided",
  "behavioral_cloning",
];

const TIER_BY_ID = {
  PPO_EASY: { name: "Easy", className: "tier-easy" },
  PPO_MEDIUM: { name: "Medium", className: "tier-medium" },
  PPO_HARD: { name: "Hard", className: "tier-hard" },
};

export default function HomePage() {
  const [loading, setLoading] = useState(false);
  const [loadingPlayers, setLoadingPlayers] = useState(true);
  const [availableOpponents, setAvailableOpponents] = useState([]);
  const [numPlayers, setNumPlayers] = useState(2);
  const [selectedOpponentIds, setSelectedOpponentIds] = useState([]);
  const history = useHistory();
  const { user, isAuthenticated, logout, loading: authLoading } = useAuth();
  const isOneVsOne = numPlayers === 2;
  const opponentSlots = numPlayers - 1;

  const selectableOpponents = useMemo(() => {
    if (isOneVsOne) return availableOpponents;
    // PPO-like model opponents are restricted to 1v1 only.
    return availableOpponents.filter((player) => player.kind !== "ppo");
  }, [availableOpponents, isOneVsOne]);

  const opponentGroups = useMemo(() => {
    return CATEGORY_ORDER.map((category) => ({
      category,
      items: selectableOpponents.filter((player) => player.category === category),
    })).filter((group) => group.items.length > 0);
  }, [selectableOpponents]);

  const defaultOpponentIds = useMemo(() => {
    if (selectableOpponents.length === 0) {
      return Array(opponentSlots).fill("");
    }

    const preferred = selectableOpponents.find((p) => p.id === "CATANATRON");
    const ordered = preferred
      ? [preferred, ...selectableOpponents.filter((p) => p.id !== preferred.id)]
      : selectableOpponents;

    return Array.from({ length: opponentSlots }, (_, index) => {
      return ordered[index % ordered.length].id;
    });
  }, [selectableOpponents, opponentSlots]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const players = await getPlayers();
        if (!mounted) return;
        setAvailableOpponents(players.filter((player) => player.id !== "HUMAN"));
      } finally {
        if (mounted) setLoadingPlayers(false);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    setSelectedOpponentIds((previous) => {
      return Array.from({ length: opponentSlots }, (_, index) => {
        const previousValue = previous[index];
        const stillValid = selectableOpponents.some(
          (player) => player.id === previousValue
        );
        return stillValid ? previousValue : defaultOpponentIds[index] || "";
      });
    });
  }, [opponentSlots, selectableOpponents, defaultOpponentIds]);

  const canStartGame =
    selectedOpponentIds.length === opponentSlots &&
    selectedOpponentIds.every((playerId) => Boolean(playerId));

  const handleOpponentChange = (slotIndex, value) => {
    setSelectedOpponentIds((previous) => {
      const next = [...previous];
      next[slotIndex] = value;
      return next;
    });
  };

  const handleCreateGame = async () => {
    if (!canStartGame) return;
    setLoading(true);
    try {
      const players = ["HUMAN", ...selectedOpponentIds];
      const gameId = await createGame(players);
      history.push("/games/" + gameId);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="home-page">
      <div className="account-strip page-top-right">
        {!authLoading && !isAuthenticated && (
          <>
            <Button
              size="small"
              variant="outlined"
              onClick={() => history.push("/login")}
            >
              Sign In
            </Button>
            <Button
              size="small"
              variant="outlined"
              onClick={() => history.push("/register")}
            >
              Register
            </Button>
          </>
        )}
        {!authLoading && isAuthenticated && (
          <>
            <Typography variant="caption" className="account-email">
              Signed in as {user?.email}
            </Typography>
            <Button size="small" variant="outlined" onClick={() => history.push("/me")}>
              My Games
            </Button>
            <Button size="small" variant="outlined" onClick={logout}>
              Sign Out
            </Button>
          </>
        )}
      </div>

      <h1 className="logo">Catanatron</h1>

      <div className="switchable">
        {!loading && !loadingPlayers ? (
          <div className="game-setup-panel">
            <Typography className="setup-title" variant="h6">
              Match Setup
            </Typography>
            <Typography className="setup-subtitle" variant="body2">
              Select player count and configure each opponent seat.
            </Typography>

            <div className="setup-inputs">
              <div className="players-picker">
                <Typography
                  variant="caption"
                  className="opponent-card-section-label"
                >
                  Players
                </Typography>
                <div className="opponent-card-row players-chips">
                  {[2, 3, 4].map((count) => (
                    <button
                      type="button"
                      key={`players-${count}`}
                      className={`opponent-chip ${
                        numPlayers === count ? "is-selected" : ""
                      }`}
                      onClick={() => setNumPlayers(count)}
                    >
                      {count} Players
                    </button>
                  ))}
                </div>
              </div>

              {isOneVsOne ? (
                <OpponentCardPicker
                  groups={opponentGroups}
                  selectedId={selectedOpponentIds[0] || ""}
                  onSelect={(playerId) => handleOpponentChange(0, playerId)}
                />
              ) : (
                <MultiOpponentPicker
                  groups={opponentGroups}
                  slotCount={opponentSlots}
                  selectedIds={selectedOpponentIds}
                  onSelect={handleOpponentChange}
                />
              )}
            </div>

            <Button
              variant="contained"
              color="primary"
              onClick={handleCreateGame}
              disabled={!canStartGame}
            >
              Start Game
            </Button>
          </div>
        ) : (
          <Loader
            className="loader"
            type="Grid"
            color="#ffffff"
            height={60}
            width={60}
          />
        )}
      </div>
    </div>
  );
}

function MultiOpponentPicker({ groups, slotCount, selectedIds, onSelect }) {
  const availablePlayers = groups.flatMap((group) => group.items);
  return (
    <div className="multi-opponent-picker">
      {Array.from({ length: slotCount }, (_, slotIndex) => (
        <div
          key={`opponent-slot-${slotIndex}`}
          className="opponent-card-section"
        >
          <Typography
            variant="caption"
            className="opponent-card-section-label"
          >
            {`Opponent ${slotIndex + 1}`}
          </Typography>
          <div className="opponent-card-row baseline-chips">
            {availablePlayers.map((player) => {
              const isSelected = selectedIds[slotIndex] === player.id;
              return (
                <button
                  type="button"
                  key={`${slotIndex}-${player.id}`}
                  className={`opponent-chip ${isSelected ? "is-selected" : ""}`}
                  onClick={() => onSelect(slotIndex, player.id)}
                  title={player.description}
                >
                  {player.label}
                </button>
              );
            })}
          </div>
        </div>
      ))}
      <Typography
        variant="caption"
        className="multi-opponent-helper"
      >
        Model-based AI opponents are available only in 1v1.
      </Typography>
    </div>
  );
}

function OpponentCardPicker({ groups, selectedId, onSelect }) {
  const tierPlayers = groups
    .filter(
      (group) =>
        group.category === "expert_guided" ||
        group.category === "behavioral_cloning"
    )
    .flatMap((group) => group.items);
  const baselinePlayers = groups
    .filter((group) => group.category === "baseline")
    .flatMap((group) => group.items);

  return (
    <div className="opponent-card-picker">
      {tierPlayers.length > 0 && (
        <div className="opponent-card-section">
          <Typography variant="caption" className="opponent-card-section-label">
            AI Opponents
          </Typography>
          <div className="opponent-card-row tier-cards">
            {tierPlayers.map((player) => {
              const tier = TIER_BY_ID[player.id] || { name: "AI", className: "" };
              const isSelected = player.id === selectedId;
              return (
                <button
                  type="button"
                  key={player.id}
                  className={`opponent-card ${tier.className} ${
                    isSelected ? "is-selected" : ""
                  }`}
                  onClick={() => onSelect(player.id)}
                >
                  <span className="opponent-card-tier">{tier.name}</span>
                  <span className="opponent-card-label">{player.label}</span>
                  <span className="opponent-card-description">
                    {player.description}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
      {baselinePlayers.length > 0 && (
        <div className="opponent-card-section">
          <Typography variant="caption" className="opponent-card-section-label">
            Baselines
          </Typography>
          <div className="opponent-card-row baseline-chips">
            {baselinePlayers.map((player) => {
              const isSelected = player.id === selectedId;
              return (
                <button
                  type="button"
                  key={player.id}
                  className={`opponent-chip ${isSelected ? "is-selected" : ""}`}
                  onClick={() => onSelect(player.id)}
                  title={player.description}
                >
                  {player.label}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
