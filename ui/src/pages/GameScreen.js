import React, { useEffect, useState, useContext } from "react";
import { useParams } from "react-router-dom";
import PropTypes from "prop-types";
import Loader from "react-loader-spinner";
import { useSnackbar } from "notistack";

import ZoomableBoard from "./ZoomableBoard";
import ActionsToolbar from "./ActionsToolbar";

import "react-loader-spinner/dist/loader/css/react-spinner-loader.css";
import "./GameScreen.scss";
import LeftDrawer from "../components/LeftDrawer";
import RightDrawer from "../components/RightDrawer";
import { store } from "../store";
import ACTIONS from "../actions";
import { getState } from "../utils/apiClient";
import { openGameSocket } from "../utils/socketClient";
import { dispatchSnackbar } from "../components/Snackbar";
import { getHumanColor } from "../utils/stateUtils";

function GameScreen({ replayMode }) {
  const { gameId, stateIndex } = useParams();
  const { state, dispatch } = useContext(store);
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();
  const [isBotThinking, setIsBotThinking] = useState(false);

  // Load game state
  useEffect(() => {
    if (!gameId) {
      return;
    }

    (async () => {
      const gameState = await getState(gameId, stateIndex);
      dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
    })();
  }, [gameId, stateIndex, dispatch]);

  // Subscribe to realtime updates.
  useEffect(() => {
    if (!gameId || replayMode) {
      return;
    }

    return openGameSocket(
      gameId,
      (gameState) => {
        setIsBotThinking(false);
        dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
        if (getHumanColor(gameState)) {
          dispatchSnackbar(enqueueSnackbar, closeSnackbar, gameState);
        }
      },
      () => {
        setIsBotThinking(false);
      }
    );
  }, [
    gameId,
    replayMode,
    dispatch,
    enqueueSnackbar,
    closeSnackbar,
  ]);

  useEffect(() => {
    if (!state.gameState || replayMode) return;
    const botTurn =
      state.gameState.bot_colors.includes(state.gameState.current_color) &&
      !state.gameState.winning_color;
    setIsBotThinking(botTurn);
  }, [state.gameState, replayMode]);

  if (!state.gameState) {
    return (
      <main>
        <Loader
          className="loader"
          type="Grid"
          color="#000000"
          height={100}
          width={100}
        />
      </main>
    );
  }

  return (
    <main>
      <h1 className="logo">Catanatron</h1>
      <ZoomableBoard replayMode={replayMode} />
      <ActionsToolbar isBotThinking={isBotThinking} replayMode={replayMode} />
      <LeftDrawer />
      <RightDrawer />
    </main>
  );
}

GameScreen.propTypes = {
  /**
   * Injected by the documentation to work in an iframe.
   * You won't need it on your project.
   */
  window: PropTypes.func,
};

export default GameScreen;
