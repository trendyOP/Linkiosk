import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

export default function Cart() {
  const navigate = useNavigate();
  const { state } = useLocation();
  const cart = state?.cart || [];

  const total = cart.reduce((s, i) => s + i.price, 0);

  return (
    <main className="container">
      <h2 className="section-title">장바구니</h2>
      {cart.length === 0 ? (
        <p className="muted">담은 메뉴가 없습니다.</p>
      ) : (
        <ul className="cart-list">
          {cart.map((it, idx) => (
            <li key={idx}>
              <span>{it.name}</span>
              <strong>{it.price.toLocaleString()}원</strong>
            </li>
          ))}
        </ul>
      )}

      <footer className="cart-actions">
        <button className="secondary" onClick={() => navigate('/menu')}>더 담기</button>
        <button className="primary">결제하기 ({total.toLocaleString()}원)</button>
      </footer>
    </main>
  );
}
