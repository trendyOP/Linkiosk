// CompletionScreen.js
import React, { useEffect } from "react";

/**
 * 결제 완료 + 번호표 안내
 * @param {number}  orderNumber  랜덤 100~999
 * @param {fn}      onRestart    처음 화면으로
 */
export default function CompletionScreen({ orderNumber, onRestart }) {
  /* 4초 뒤 자동 초기화 */
  useEffect(() => {
    const t = setTimeout(onRestart, 4000);
    return () => clearTimeout(t);
  }, [onRestart]);

  return (
    <div className="modal-overlay">
      <div className="modal complete">
        <h2>결제가 완료되었습니다</h2>
        <p className="msg">
          메뉴가 준비되면 주문번호 호출모니터로 안내해 드립니다.
          <br />
          감사합니다
        </p>

        <button className="order-pill">{`주문번호 ${orderNumber}`}</button>
      </div>
    </div>
  );
}
