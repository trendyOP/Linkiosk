// src/components/ModeFab.js
import React from "react";
import { FaUserAlt, FaUser } from "react-icons/fa";

export default function ModeFab({ mode, onToggle }) {
  if (!onToggle) return null;
  const isElder = mode === 'elder';
  return (
    <button
      className={`mode-fab ${isElder ? 'elder' : 'normal'}`}
      onClick={() => onToggle(isElder ? 'normal' : 'elder')}
      aria-label={isElder ? '일반 모드로 전환' : '고령자 모드로 전환'}
      title={isElder ? '일반 모드로 전환' : '고령자 모드로 전환'}
    >
      {isElder ? <FaUser size={22} /> : <FaUserAlt size={22} />}
      <span className="txt">{isElder ? '일반으로' : '고령자로'}</span>
    </button>
  );
}
