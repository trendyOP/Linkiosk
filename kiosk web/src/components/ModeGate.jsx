import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useMode } from '../context/ModeContext';

export default function ModeGate() {
  const { setMode } = useMode();
  const navigate = useNavigate();

  const choose = (m) => {
    setMode(m);
    navigate('/start');
  };

  return (
    <main className="container gate">
      <h1 className="sr-only">사용 모드 선택</h1>
      <section className="gate-grid">
        <button className="gate-card normal" onClick={() => choose('normal')}>
          <h2>일반 모드</h2>
          <p>한 화면에 많은 메뉴, 빠른 탐색</p>
        </button>
        <button className="gate-card elder" onClick={() => choose('elder')}>
          <h2>고령자 모드</h2>
          <p>큰 글자·큰 버튼, 단계별 진행</p>
        </button>
      </section>
      <p className="gate-help">언제든지 화면 상단에서 모드를 바꿀 수 있어요.</p>
    </main>
  );
}
