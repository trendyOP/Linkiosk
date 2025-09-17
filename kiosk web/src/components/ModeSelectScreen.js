// src/components/ModeSelectScreen.js
import React from "react";
import { FaUser, FaUserAlt } from "react-icons/fa";

export default function ModeSelectScreen({ mode, onSelect }) {
  return (
    <div className="mode-select-screen">
      <h1 className="title">사용 모드를 선택해 주세요</h1>
      <p className="subtitle">언제든 바꿀 수 있어요. (다음 화면에서 매장/포장을 선택합니다)</p>

      <div className="mode-grid">
        <button className={`mode-card ${mode==='normal' ? 'active' : ''}`} onClick={() => onSelect('normal')}>
          <FaUser size={64} />
          <div className="label">일반 모드</div>
          <div className="desc">여러 항목을 한눈에 보기</div>
        </button>

        <button className={`mode-card ${mode==='elder' ? 'active' : ''}`} onClick={() => onSelect('elder')}>
          <FaUserAlt size={64} />
          <div className="label">고령자 모드</div>
          <div className="desc">큰 글씨, 큰 버튼, 한번에 하나씩</div>
        </button>
      </div>
    </div>
  );
}