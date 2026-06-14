import React from "react";
import { MemoryRouter, Route } from "react-router-dom";
import { render, screen } from "@testing-library/react";
import { SnackbarProvider } from "notistack";

import GameScreen from "./GameScreen";
import { StateProvider } from "../store";
import { getState } from "../utils/apiClient";
import { openGameSocket } from "../utils/socketClient";

jest.mock("./ZoomableBoard", () => () => <div>Mock Board</div>);
jest.mock("./ActionsToolbar", () => () => <div>Mock Toolbar</div>);
jest.mock("../components/LeftDrawer", () => () => <div>Mock Left Drawer</div>);
jest.mock("../components/RightDrawer", () => () => <div>Mock Right Drawer</div>);

jest.mock("../utils/apiClient", () => ({
  getState: jest.fn(),
}));

jest.mock("../utils/socketClient", () => ({
  openGameSocket: jest.fn(),
}));

describe("GameScreen realtime subscription", () => {
  it("subscribes to socket updates and disconnects on unmount", async () => {
    getState.mockResolvedValue({
      bot_colors: [],
      current_color: "RED",
      winning_color: null,
    });
    const disconnectMock = jest.fn();
    openGameSocket.mockReturnValue(disconnectMock);

    const { unmount } = render(
      <SnackbarProvider>
        <StateProvider>
          <MemoryRouter initialEntries={["/games/abc123"]}>
            <Route path="/games/:gameId">
              <GameScreen replayMode={false} />
            </Route>
          </MemoryRouter>
        </StateProvider>
      </SnackbarProvider>
    );

    await screen.findByText("Mock Board");
    expect(getState).toHaveBeenCalledWith("abc123", undefined);
    expect(openGameSocket).toHaveBeenCalledWith(
      "abc123",
      expect.any(Function),
      expect.any(Function)
    );

    unmount();
    expect(disconnectMock).toHaveBeenCalledTimes(1);
  });
});
