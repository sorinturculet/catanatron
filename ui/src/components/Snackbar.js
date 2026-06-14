import React from "react";
import { IconButton } from "@material-ui/core";
import CloseIcon from "@material-ui/icons/Close";
import { humanizeAction } from "./Prompt";

export const snackbarActions = (closeSnackbar) => (key) =>
  (
    <>
      <IconButton
        size="small"
        aria-label="close"
        color="inherit"
        onClick={() => closeSnackbar(key)}
      >
        <CloseIcon fontSize="small" />
      </IconButton>
    </>
  );

export function dispatchSnackbar(enqueueSnackbar, closeSnackbar, gameState) {
  const actions = Array.isArray(gameState?.actions) ? gameState.actions : [];
  const latestAction = actions[actions.length - 1];
  if (!Array.isArray(latestAction) || latestAction.length < 2) {
    return false;
  }

  enqueueSnackbar(humanizeAction(gameState, latestAction), {
    action: snackbarActions(closeSnackbar),
    onClick: () => {
      closeSnackbar();
    },
  });
  return true;
}
