import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import {
  clearAuthToken,
  getMe,
  getStoredToken,
  loginAccount,
  registerAccount,
  setAuthToken,
} from "./utils/apiClient";

const AuthContext = createContext({
  user: null,
  loading: true,
  isAuthenticated: false,
  login: async () => {},
  register: async () => {},
  logout: () => {},
  refreshUser: async () => {},
});

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const refreshUser = async () => {
    const me = await getMe();
    setUser(me);
    return me;
  };

  useEffect(() => {
    const token = getStoredToken();
    if (!token) {
      setLoading(false);
      return;
    }

    (async () => {
      try {
        await refreshUser();
      } catch (error) {
        clearAuthToken();
        setUser(null);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const login = async (email, password) => {
    const payload = await loginAccount(email, password);
    setAuthToken(payload.access_token);
    setUser(payload.user);
    return payload.user;
  };

  const register = async (email, password) => {
    const payload = await registerAccount(email, password);
    setAuthToken(payload.access_token);
    setUser(payload.user);
    return payload.user;
  };

  const logout = () => {
    clearAuthToken();
    setUser(null);
  };

  const value = useMemo(
    () => ({
      user,
      loading,
      isAuthenticated: Boolean(user),
      login,
      register,
      logout,
      refreshUser,
    }),
    [user, loading]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}
