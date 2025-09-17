// src/components/StartModeScreen.js
import React from "react";

export default function StartModeScreen({ onSelect }){
  return (
    <div className="centered startmode">
      <h1 className="title">이용 모드를 선택해 주세요</h1>
      <p className="desc">한 손으로 쉽게: 고령자 모드는 한 번에 하나씩 안내합니다.</p>
      <div className="btn-row">
        <button className="btn-primary xxl" onClick={() => onSelect("elder")}>
          고령자 모드 시작
        </button>
        <button className="btn-outline xxl" onClick={() => onSelect("normal")}>
          일반 모드 시작
        </button>
      </div>
    </div>
  );
}
