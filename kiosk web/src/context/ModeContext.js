import { createContext, useContext, useEffect, useMemo, useState } from "react";

const ModeContext = createContext();

export function ModeProvider({ children }) {
  const [mode, setMode] = useState(localStorage.getItem("mode") || "normal"); // 'elder' | 'normal'
  const [fontScale, setFontScale] = useState(localStorage.getItem("fontScale") || "lg"); // 'lg' | 'xl'

  useEffect(() => {
    document.documentElement.dataset.mode = mode;
    document.documentElement.dataset.font = fontScale;
    localStorage.setItem("mode", mode);
    localStorage.setItem("fontScale", fontScale);
  }, [mode, fontScale]);

  const value = useMemo(
    () => ({
      mode,
      toNormal: () => setMode("normal"),
      toElder: () => setMode("elder"),
      fontScale,
      setFontScale,
      bigger: () => setFontScale((v) => (v === "lg" ? "xl" : "xl")),
      smaller: () => setFontScale("lg"),
    }),
    [mode, fontScale]
  );

  return <ModeContext.Provider value={value}>{children}</ModeContext.Provider>;
}

export function useMode() {
  return useContext(ModeContext);
}
