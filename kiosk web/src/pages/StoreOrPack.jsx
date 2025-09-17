import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useMode } from '../context/ModeContext';

export default function StoreOrPack() {
  const navigate = useNavigate();
  const { mode } = useMode();

  const goMenu = () => navigate('/menu');

  return (
    <main className="container">
      <h2 className="section-title">이용 유형을 선택하세요</h2>
      <div className={`choice-grid ${mode === 'elder' ? 'choice-grid-elder' : ''}`}>
        <button className="choice" onClick={goMenu}>
          <span className="emoji" aria-hidden>🏠</span>
          <span>매장</span>
        </button>
        <button className="choice" onClick={goMenu}>
          <span className="emoji" aria-hidden>📦</span>
          <span>포장</span>
        </button>
      </div>
      <p className="muted">처음에만 정하면 됩니다. 다음 화면에서 메뉴를 고르세요.</p>
    </main>
  );
}
