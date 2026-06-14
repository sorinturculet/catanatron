import axios from "axios";

import { API_URL } from "../configuration";

const TOKEN_KEY = "catanatron_jwt";

export const http = axios.create({
  baseURL: API_URL,
});

export function getStoredToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setAuthToken(token) {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
    http.defaults.headers.common.Authorization = `Bearer ${token}`;
  } else {
    localStorage.removeItem(TOKEN_KEY);
    delete http.defaults.headers.common.Authorization;
  }
}

export function clearAuthToken() {
  setAuthToken(null);
}

const initialToken = getStoredToken();
if (initialToken) {
  setAuthToken(initialToken);
}

http.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      clearAuthToken();
    }
    return Promise.reject(error);
  }
);

export async function registerAccount(email, password) {
  const response = await http.post("/api/auth/register", { email, password });
  return response.data;
}

export async function loginAccount(email, password) {
  const response = await http.post("/api/auth/login", { email, password });
  return response.data;
}

export async function getMe() {
  const response = await http.get("/api/auth/me");
  return response.data;
}

export async function getMyGames(page = 1, perPage = 20) {
  const response = await http.get("/api/me/games", {
    params: { page, per_page: perPage },
  });
  return response.data;
}

export async function forgotPassword(email) {
  const response = await http.post("/api/auth/forgot-password", { email });
  return response.data;
}

export async function resetPassword(token, password) {
  const response = await http.post("/api/auth/reset-password", { token, password });
  return response.data;
}

export async function getPlayers() {
  const response = await http.get("/api/players");
  return response.data;
}

export async function createGame(players) {
  const response = await http.post("/api/games", { players });
  return response.data.game_id;
}

export async function getState(gameId, stateIndex = "latest") {
  const response = await http.get(`/api/games/${gameId}/states/${stateIndex}`);
  return response.data;
}

/** action=undefined means bot action */
export async function postAction(gameId, action = undefined) {
  const response = await http.post(`/api/games/${gameId}/actions`, action);
  return response.data;
}

export async function getMctsAnalysis(gameId, stateIndex = "latest") {
  try {
    console.log("Getting MCTS analysis for:", {
      gameId,
      stateIndex,
      url: `${API_URL}/api/games/${gameId}/states/${stateIndex}/mcts-analysis`,
    });

    if (!gameId) {
      throw new Error("No gameId provided to getMctsAnalysis");
    }

    const response = await http.get(
      `/api/games/${gameId}/states/${stateIndex}/mcts-analysis`
    );

    console.log("MCTS analysis response:", response.data);
    return response.data;
  } catch (error) {
    console.error("MCTS analysis error:", {
      message: error.message,
      status: error.response?.status,
      data: error.response?.data,
      stack: error.stack,
    });
    throw error;
  }
}
