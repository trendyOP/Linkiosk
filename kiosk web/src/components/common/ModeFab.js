// src/components/common/ModeFab.js
import React from "react";

/** 고정 위치 모드 전환 버튼 (일반 ↔ 고령자) */
export default function ModeFab({ elder=false, onToggle }){
  const label = elder ? "일반 모드" : "고령자 모드";
  return (
    <button
      className="mode-fab"
      aria-label={`모드 전환: ${label}`}
      onClick={()=> onToggle(!elder)}
      title={label}
    >
      {label}
    </button>
  );
}
