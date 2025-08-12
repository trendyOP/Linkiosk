// CardPaymentScreen.js
import React, { useEffect, useState } from "react";

export default function CardPaymentScreen({ amount, onComplete }) {
  const [step, setStep] = useState(1);            // 1 입력  → 2 영수증 여부
  const [timer, setTimer] = useState(2);

  /* 2초 뒤 카드 입력창 자동 종료 */
  useEffect(() => {
    if (step === 1) {
      const t = setTimeout(() => setStep(2), 2000);
      return () => clearTimeout(t);
    }
  }, [step]);

  /* 영수증 여부 → 완료 */
  const handleReceipt = () => {
    onComplete();
  };

  return (
    <div className="modal-overlay">
      <div className="modal card">
        {/* ── 헤더 ── */}
        <h2 className="card-head">신용카드 결제</h2>
        <p className="card-warn">결제가 완료될때까지 카드를 빼지 마세요!</p>

        {step === 1 && (
          <>
            {/* 카드 입력 UI 더미 */}
            <table className="card-form">
              <tbody>
                <tr><th>카드번호</th><td>333333******333*</td></tr>
                <tr><th>카드사</th><td>**비자개인</td></tr>
                <tr><th>유효기간</th><td> 년  월</td></tr>
                <tr><th>할부개월수</th><td> </td></tr>
                <tr><th>결제금액</th><td>{amount.toLocaleString()}</td></tr>
              </tbody>
            </table>

            <div className="numpad-dummy" />

            <div className="slot" />
            <p className="slot-text">카드 정보 요청 중입니다</p>
          </>
        )}

        {step === 2 && (
          <>
            <p className="receipt-q">영수증 출력을 하시겠습니까?</p>
            <div className="yesno">
              <button className="btn-primary" onClick={handleReceipt}>예</button>
              <button className="btn-outline" onClick={handleReceipt}>아니오</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
