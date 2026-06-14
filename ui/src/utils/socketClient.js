import { io } from "socket.io-client";

import { API_URL } from "../configuration";

export function openGameSocket(gameId, onStateUpdated, onGameCompleted) {
  const socket = io(API_URL, { transports: ["websocket", "polling"] });

  socket.on("connect", () => {
    socket.emit("subscribe", { game_id: gameId });
  });
  socket.on("state_updated", onStateUpdated);
  socket.on("game_completed", onGameCompleted);

  return () => {
    socket.disconnect();
  };
}
