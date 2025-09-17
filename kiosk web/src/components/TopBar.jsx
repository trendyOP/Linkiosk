import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ModeToggleButton from './ModeToggleButton';
import { useMode } from '../context/ModeContext';

export default function TopBar() {
  const navigate = useNavigate();
  const location = useLocation();
  const { mode } = useMode();

  const titleMap = {
    '/mode': '사용 모드 선택',
    '/start': '매장/포장 선택',
    '/menu': '메뉴',
    '/cart': '장바구니',
  };
  const title = titleMap[location.pathname] || 'LINKIOSK';

  return (
    <header className="topbar" role="banner">
      <div className="left">
        {location.pathname !== '/mode' && (
          <button className="back" onClick={() => navigate(-1)} aria-label="뒤로가기">←</button>
        )}
        <strong className="logo">LINKIOSK</strong>
      </div>
      <div className="center"><h1>{title}</h1></div>
      <div className="right">
        <span className="badge" data-variant={mode}>{mode === 'elder' ? '고령자' : '일반'}</span>
        <ModeToggleButton />
      </div>
    </header>
  );
}
