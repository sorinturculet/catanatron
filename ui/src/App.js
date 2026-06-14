import React from "react";
import { BrowserRouter as Router, Switch, Route, Redirect } from "react-router-dom";
import { SnackbarProvider } from "notistack";
import { createTheme, ThemeProvider } from "@material-ui/core/styles";
import { blue, green } from "@material-ui/core/colors";
import Fade from "@material-ui/core/Fade";
import Loader from "react-loader-spinner";

import GameScreen from "./pages/GameScreen";
import HomePage from "./pages/HomePage";
import { StateProvider } from "./store";
import { AuthProvider, useAuth } from "./auth";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import DashboardPage from "./pages/DashboardPage";
import ForgotPasswordPage from "./pages/ForgotPasswordPage";
import ResetPasswordPage from "./pages/ResetPasswordPage";

import "./App.scss";

const theme = createTheme({
  palette: {
    primary: {
      main: blue[900],
    },
    secondary: {
      main: green[900],
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <AuthProvider>
        <StateProvider>
          <SnackbarProvider
            classes={{ containerRoot: ["snackbar-container"] }}
            maxSnack={1}
            autoHideDuration={1000}
            TransitionComponent={Fade}
            TransitionProps={{ timeout: 100 }}
          >
            <Router>
              <Switch>
                <Route path="/games/:gameId/states/:stateIndex">
                  <GameScreen replayMode={true} />
                </Route>
                <Route path="/games/:gameId">
                  <GameScreen replayMode={false} />
                </Route>
                <Route path="/login" exact={true}>
                  <LoginPage />
                </Route>
                <Route path="/register" exact={true}>
                  <RegisterPage />
                </Route>
                <Route path="/forgot-password" exact={true}>
                  <ForgotPasswordPage />
                </Route>
                <Route path="/reset-password" exact={true}>
                  <ResetPasswordPage />
                </Route>
                <ProtectedRoute path="/me" exact={true}>
                  <DashboardPage />
                </ProtectedRoute>
                <Route path="/" exact={true}>
                  <HomePage />
                </Route>
              </Switch>
            </Router>
          </SnackbarProvider>
        </StateProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}

function ProtectedRoute({ children, ...rest }) {
  const { loading, isAuthenticated } = useAuth();
  return (
    <Route
      {...rest}
      render={({ location }) => {
        if (loading) {
          return (
            <main className="center-loader-screen">
              <Loader type="Grid" color="#ffffff" height={60} width={60} />
            </main>
          );
        }
        if (!isAuthenticated) {
          return (
            <Redirect
              to={{ pathname: "/login", state: { from: location.pathname } }}
            />
          );
        }
        return children;
      }}
    />
  );
}

export default App;
