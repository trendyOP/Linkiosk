import React, { useState, useEffect, useRef } from "react";
import { FaHome, FaShoppingCart, FaChevronLeft, FaChevronRight } from "react-icons/fa";
import { categories } from "../data/menuData";

// ── 메뉴 썸네일 이미지 import (320 & 600 해상도) ──
const importAll = (r) =>
  r.keys().reduce((acc, key) => {
    const match = key.match(/\.\/(.+)-(\d+)\.(png|jpe?g)$/);
    if (match) {
      const [, name, size] = match;
      acc[name] = acc[name] || {};
      acc[name][size] = r(key);
    }
    return acc;
  }, {});
const images = importAll(
  require.context("../assets/images", false, /\.(png|jpe?g)$/)
);

export default function MenuScreen({
  currentCategory,
  onCategoryChange,
  onGoHome,
  onSelectItem,
  cart = [],
  onViewCart,
  onCheckout
}) {
  const [categoryId, setCategoryId] = useState(currentCategory);
  const catBarRef = useRef(null);

  useEffect(() => {
    setCategoryId(currentCategory);
  }, [currentCategory]);

  const changeCat = (id) => {
    setCategoryId(id);
    onCategoryChange?.(id);
  };

  const scrollCats = (offset) => {
    if (catBarRef.current) {
      catBarRef.current.scrollBy({ left: offset, behavior: "smooth" });
    }
  };

  const category = categories.find((c) => c.id === categoryId) || categories[0];
  const items = category.items;

  const totalQty = cart.reduce((sum, i) => sum + (i.qty || 1), 0);
  const totalPrice = cart.reduce((sum, i) => sum + (i.total || i.price || 0), 0);

  return (
    <div className="menu-screen">
      {/* 헤더 */}
      <header className="app-header">
        <button className="home-btn" onClick={onGoHome}>
          <FaHome size={22} />
        </button>
        <h1 className="app-title">카페</h1>
      </header>

      {/* 카테고리 + 화살표 네비 */}
      <div className="category-nav">
        <button
          className="cat-nav-btn left"
          onClick={() => scrollCats(-200)}
          aria-label="이전 카테고리"
        >
          <FaChevronLeft size={18} />
        </button>

        <nav ref={catBarRef} className="category-bar">
          {categories.map((c) => (
            <div
              key={c.id}
              className={`category-item ${c.id === categoryId ? "active" : ""}`}
              onClick={() => changeCat(c.id)}
            >
              {c.name}
            </div>
          ))}
        </nav>

        <button
          className="cat-nav-btn right"
          onClick={() => scrollCats(200)}
          aria-label="다음 카테고리"
        >
          <FaChevronRight size={18} />
        </button>
      </div>

      {/* 메뉴 리스트 (스크롤링) */}
      <main className="menu-body">
        <div className="menu-grid">
          {items.map((m) => {
            const src320 = images[m.imageKey]?.["320"];
            const src600 = images[m.imageKey]?.["600"];
            return (
              <article
                key={m.id}
                className="menu-item"
                onClick={() => onSelectItem?.({ ...m, category: category.id })}
              >
                <div className="thumb">
                  {src320 && src600 && (
                    <img
                      src={src320}
                      srcSet={`${src320} 320w, ${src600} 600w`}
                      sizes="(max-width:480px) 100vw, (min-width:1440px) calc((100vw-32px*2)/8), 160px"
                      alt={m.name}
                      className="thumb-img"
                    />
                  )}
                </div>
                <p className="name">{m.name}</p>
                <p className="price">{m.price.toLocaleString()}원</p>
              </article>
            );
          })}
        </div>
      </main>

      {/* 하단 결제바 */}
      <footer className="checkout-bar">
        <button className="cart-link" onClick={onViewCart}>
          <FaShoppingCart size={18} /> 장바구니 보기
          {totalQty > 0 && <span className="badge">{totalQty}</span>}
        </button>
        <button className="btn-pay" onClick={onCheckout}>
          총 {totalQty}개 {totalPrice.toLocaleString()}원 결제하기
        </button>
      </footer>
    </div>
);
}