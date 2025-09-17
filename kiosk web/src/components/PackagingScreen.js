// PackagingScreen.js
import React from 'react';
import { FaUtensils, FaShoppingBag } from 'react-icons/fa';

export default function PackagingScreen({ onNext, elderMode, setElderMode }) {
  return (
    <div className="packaging-screen">
      <div className="mode-toggle" role="group" aria-label="화면 모드 선택">
        <button className={`mode-btn ${!elderMode ? 'on' : ''}`} aria-pressed={!elderMode} onClick={() => setElderMode(false)}>일반 모드</button>
        <button className={`mode-btn ${elderMode ? 'on' : ''}`} aria-pressed={elderMode} onClick={() => setElderMode(true)}>고령자 모드</button>
      </div>
    
      <h1 className="packaging-header">포장 선택</h1>
      <div className="packaging-divider" />

      <div className="packaging-options">
        <button className="packaging-option" onClick={() => onNext('eat-in')}>
          <FaUtensils size={48} />
          <span>매장식사</span>
        </button>

        <button className="packaging-option" onClick={() => onNext('take-out')}>
          <FaShoppingBag size={48} />
          <span>포장</span>
        </button>
      </div>
    </div>
  );
}
