import { dispatchSnackbar } from "./Snackbar";

describe("dispatchSnackbar", () => {
  it("skips snackbar when there is no latest action", () => {
    const enqueueSnackbar = jest.fn();
    const closeSnackbar = jest.fn();

    const didDispatch = dispatchSnackbar(enqueueSnackbar, closeSnackbar, {
      actions: [],
      bot_colors: ["BLUE"],
    });

    expect(didDispatch).toBe(false);
    expect(enqueueSnackbar).not.toHaveBeenCalled();
  });

  it("dispatches snackbar when an action exists", () => {
    const enqueueSnackbar = jest.fn();
    const closeSnackbar = jest.fn();

    const didDispatch = dispatchSnackbar(enqueueSnackbar, closeSnackbar, {
      actions: [["BLUE", "END_TURN"]],
      bot_colors: ["BLUE"],
    });

    expect(didDispatch).toBe(true);
    expect(enqueueSnackbar).toHaveBeenCalledTimes(1);
  });
});
