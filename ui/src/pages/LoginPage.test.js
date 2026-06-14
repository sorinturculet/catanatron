import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";

import LoginPage from "./LoginPage";
import { useAuth } from "../auth";

const mockPush = jest.fn();
const mockReplace = jest.fn();

jest.mock("../auth", () => ({
  useAuth: jest.fn(),
}));

jest.mock("react-router-dom", () => ({
  ...jest.requireActual("react-router-dom"),
  useHistory: () => ({ push: mockPush, replace: mockReplace }),
  useLocation: () => ({ state: {} }),
}));

describe("LoginPage", () => {
  beforeEach(() => {
    mockPush.mockReset();
    mockReplace.mockReset();
    useAuth.mockReturnValue({
      login: jest.fn(),
      isAuthenticated: false,
    });
  });

  test("forgot password link routes to forgot-password page", () => {
    render(<LoginPage />);

    fireEvent.click(screen.getByText("Reset it"));
    expect(mockPush).toHaveBeenCalledWith("/forgot-password");
  });
});
