// PaymentMethodScreen.js
import React, { useState, useMemo } from "react";
import { FaMobileAlt } from "react-icons/fa";            // ← 아이콘

/**
 * 결제수단 선택 + 주문내역 확인 모달
 * @param {Array}  cart   [{name, qty, total, packaging, options}]
 * @param {fn}     onBack 모달 닫기
 * @param {fn}     onPay  결제수단 확정 → 다음 단계
 */
export default function PaymentMethodScreen({ cart = [], onBack, onPay }) {
  const [method, setMethod] = useState("voucher");

  const { totalQty, totalPrice } = useMemo(() => {
    const qty   = cart.reduce((s, i) => s + i.qty,   0);
    const price = cart.reduce((s, i) => s + i.total, 0);
    return { totalQty: qty, totalPrice: price };
  }, [cart]);

  return (
    <div className="modal-overlay">
      <div className="modal pay">
        {/* ─── 헤더 ─── */}
        <header className="pay-head">
          <h2>주문내역확인</h2>
          <button className="btn-close" onClick={onBack}>✕</button>
        </header>

        {/* ─── 결제수단 ─── */}
        <section className="pay-method">
          <h3>결제할 기능을 선택하세요</h3>
          <div className="methods">
            <button
              className={`method ${method === "voucher" ? "active" : ""}`}
              onClick={() => setMethod("voucher")}
            >
              <FaMobileAlt size={48} />
              <span>모바일 상품권</span>
            </button>
            {/* 필요 시 다른 수단을 이곳에 추가 */}
          </div>
        </section>

        {/* ─── 주문정보 ─── */}
        <section className="pay-order">
          <h3>주문정보를 확인해 주세요</h3>
          <table>
            <thead>
              <tr><th>상품명</th><th>수량</th><th>가격</th></tr>
            </thead>
            <tbody>
              {cart.map((c, i) => (
                <tr key={i}>
                  <td>
                    [{c.packaging === "take-out" ? "포장" : "매장"}]
                    {c.name}
                    {Object.values(c.options ?? {})
                      .flat()
                      .map((o) => ` ▶ ${o.name}`)
                      .join(" ")}
                  </td>
                  <td>{c.qty}</td>
                  <td>{c.total.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        {/* ─── 합계 ─── */}
        <section className="pay-summary">
          <div>
            총 수량&nbsp; <strong>{totalQty}</strong>개&nbsp;&nbsp;
            주문금액&nbsp; <strong>{totalPrice.toLocaleString()}</strong>원
          </div>
          <div>할인금액&nbsp; 0원&nbsp;&nbsp; 결제한 금액&nbsp; 0원</div>
          <div className="to-pay">
            결제할 금액&nbsp; <strong>{totalPrice.toLocaleString()}원</strong>
          </div>
        </section>

        {/* ─── 확인 버튼 ─── */}
        <footer className="pay-foot">
          <button
            className="btn-primary"
            onClick={() => onPay(method)}
            disabled={!method}
            onPay={() => setStep(7)}
          >
            확인
          </button>
        </footer>
      </div>
    </div>
  );
}
