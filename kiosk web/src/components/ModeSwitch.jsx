import { useMode } from "../context/ModeContext";

export default function ModeSwitch() {
  const { mode, toNormal, toElder, fontScale, setFontScale } = useMode();
  return (
    <div className="mode-switch" style={{display:'flex', gap:12, padding:12}}>
      <button onClick={toNormal} aria-pressed={mode === "normal"}>일반</button>
      <button onClick={toElder} aria-pressed={mode === "elder"}>고령자</button>
      <button onClick={() => setFontScale(fontScale === "lg" ? "xl" : "lg")}>
        글자 {fontScale === "lg" ? "더 크게" : "기본 크기"}
      </button>
    </div>
  );
}
