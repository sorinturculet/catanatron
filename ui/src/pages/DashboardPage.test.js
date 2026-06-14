import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";

import DashboardPage from "./DashboardPage";
import { useAuth } from "../auth";
import { getMyGames } from "../utils/apiClient";

const mockPush = jest.fn();

jest.mock("../auth", () => ({
  useAuth: jest.fn(),
}));

jest.mock("../utils/apiClient", () => ({
  getMyGames: jest.fn(),
}));

jest.mock("react-router-dom", () => ({
  ...jest.requireActual("react-router-dom"),
  useHistory: () => ({ push: mockPush }),
}));

describe("DashboardPage", () => {
  beforeEach(() => {
    mockPush.mockReset();
    getMyGames.mockReset();
    useAuth.mockReturnValue({
      user: { email: "player@example.com" },
      logout: jest.fn(),
    });
  });

  test("renders game cards with smart game labels", async () => {
    getMyGames.mockResolvedValue({
      items: [
        {
          game_id: "6a158057-febc-4c08-8f33-86876ba2f1fa",
          created_at: "2026-05-13T10:00:00",
          players_config: ["HUMAN", "PPO_HARD"],
        },
      ],
    });

    render(<DashboardPage />);

    expect(await screen.findByText("Game #6A158057")).toBeInTheDocument();
    expect(screen.getByText("HUMAN vs PPO_HARD")).toBeInTheDocument();
  });

  test("open button navigates to interactive game route", async () => {
    getMyGames.mockResolvedValue({
      items: [
        {
          game_id: "11111111-2222-3333-4444-555555555555",
          created_at: "2026-05-13T10:00:00",
          players_config: ["HUMAN", "RANDOM"],
        },
      ],
    });

    render(<DashboardPage />);

    const openButton = await screen.findByRole("button", { name: /^open$/i });
    fireEvent.click(openButton);
    expect(mockPush).toHaveBeenCalledWith(
      "/games/11111111-2222-3333-4444-555555555555"
    );
  });
});
