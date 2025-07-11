// PackagingScreen.js
import React from 'react';
import { FaUtensils, FaShoppingBag } from 'react-icons/fa';

export default function PackagingScreen({ onNext }) {
  return (
    <div className="packaging-screen">
      <h1 className="packaging-header">포장 선택</h1>
      <div className="packaging-divider" />

      <div className="packaging-options">
        <button className="packaging-option" onClick={() => onNext('eat-in')}>
          <FaUtensils size={48} />
          <span>매장식사</span>
        </button>

        <button className="packaging-option" onClick={() => onNext('take-out')}>
          <FaShoppingBag size={48} />
          <span>포장</span>
        </button>
      </div>
    </div>
  );
}
