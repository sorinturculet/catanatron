import React from "react";
import { render, screen } from "@testing-library/react";

import App from "./App";

jest.mock("./pages/HomePage", () => () => <div>Home Page</div>);
jest.mock("./pages/GameScreen", () => () => <div>Game Screen</div>);
jest.mock("./pages/LoginPage", () => () => <div>Login Page</div>);
jest.mock("./pages/RegisterPage", () => () => <div>Register Page</div>);
jest.mock("./pages/ForgotPasswordPage", () => () => <div>Forgot Password Page</div>);
jest.mock("./pages/ResetPasswordPage", () => () => <div>Reset Password Page</div>);
jest.mock("./pages/DashboardPage", () => () => <div>Dashboard Page</div>);

test("renders home route", () => {
  render(<App />);
  expect(screen.getByText("Home Page")).toBeInTheDocument();
});
