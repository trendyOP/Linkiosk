// PaymentOptionScreen.js
import React from 'react';
import { FaCreditCard } from 'react-icons/fa';

/**
 * 결제수단 선택 팝업
 * @param {number} amount   결제 금액
 * @param {fn}     onCancel 닫기
 * @param {fn}     onSelect 신용카드 선택 후 → onSelect('card')
 */
export default function PaymentOptionScreen({ amount = 0, onCancel, onSelect }) {
  return (
    <div className="modal-overlay">
      <div className="popup pay-option">
        <h3>결제하실 방법을 터치해 주세요</h3>

        {/* ── 신용카드 버튼 ── */}
        <button className="card-btn" onClick={() => onSelect('card')}>
          <FaCreditCard size={64} />
          <span>신용카드</span>
        </button>

        {/* ── 금액 Breakdown ── */}
        <div className="breakdown">
          {/* 라벨 행 */}
          <div className="col label">주문금액</div>
          <div className="col label">－</div>
          <div className="col label">할인금액</div>
          <div className="col label">－</div>
          <div className="col label">결제한 금액</div>
          <div className="col label">＝</div>
          <div className="col label">결제금액</div>

          {/* 값 행 */}
          <div className="col value red">{amount.toLocaleString()}</div>
          <div className="col value" />
          <div className="col value red">0</div>
          <div className="col value" />
          <div className="col value red">0</div>
          <div className="col value" />
          <div className="col value red">{amount.toLocaleString()}</div>
        </div>

        <button className="btn-cancel" onClick={onCancel}>취소</button>
      </div>
    </div>
  );
}
