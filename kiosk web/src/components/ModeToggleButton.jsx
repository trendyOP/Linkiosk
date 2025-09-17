import React from 'react';
import { useMode } from '../context/ModeContext';

export default function ModeToggleButton() {
  const { mode, setMode } = useMode();
  const other = mode === 'elder' ? 'normal' : 'elder';

  return (
    <button
      type="button"
      className="mode-toggle"
      onClick={() => setMode(other)}
      aria-label={mode === 'elder' ? '일반 모드로 전환' : '고령자 모드로 전환'}
    >
      {mode === 'elder' ? '일반 보기' : '고령자 보기'}
    </button>
  );
}
