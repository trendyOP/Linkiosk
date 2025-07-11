// CartScreen.js
import React from "react";

export default function CartScreen({ cart, onBack, onNext, onClear, onRemove }) {
  const totalQty   = cart.reduce((s, i) => s + i.qty,   0);
  const totalPrice = cart.reduce((s, i) => s + i.total, 0);

  return (
    <div className="modal-overlay">
      <div className="modal cart">
        {/* ── 헤더 ── */}
        <div className="modal-head">
          <h2>장바구니</h2>
          <button className="link-clear" onClick={onClear}>🗑 장바구니 비우기</button>
        </div>

        {/* ── 아이템 목록 ── */}
        <div className="cart-list">
          {cart.map((c, idx) => {
            const optionTxt = Object.values(c.options || {})
              .flat()
              .map((o) => o.name)
              .join(", ");

            return (
              <div key={idx} className="cart-item">
                <button className="btn-remove" onClick={() => onRemove && onRemove(idx)}>✕</button>

                <div className="cart-info">
                  <strong>{c.name}</strong>
                  {/* 가격 + 공백 + 옵션 */}
                  <span>
                    {c.total.toLocaleString()}원
                    {optionTxt ? " " + optionTxt : ""}
                  </span>
                </div>

                <div className="cart-qty">{c.qty}</div>
              </div>
            );
          })}
        </div>

        {/* ── 푸터 ── */}
        <div className="modal-foot">
          <button className="btn-outline" onClick={onBack}>더 담기</button>
          <button className="btn-primary" onClick={onNext}>
            총 {totalQty}개 상품&nbsp; {totalPrice.toLocaleString()}원 결제하기
          </button>
        </div>
      </div>
    </div>
  );
}
