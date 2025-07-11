// src/components/MenuScreen.js
import React, { useState, useEffect, useMemo, useRef } from "react";
import { FaHome, FaShoppingCart, FaChevronLeft, FaChevronRight } from "react-icons/fa";
import { categories } from "../data/menuData";

// ─ require.context로 이미지 일괄 로드 (320 & 600 해상도) ─
const importAll = (r) =>
  r.keys().reduce((acc, key) => {
    // key 예: "./cof_ame-320.jpg" or "./cof_ame-600.jpg"
    const [, name, size] = key.match(/\.\/(.+)-(\d+)\.(png|jpe?g)$/);
    acc[name] = acc[name] || {};
    acc[name][size] = r(key);
    return acc;
  }, {});

const images = importAll(
  require.context("../assets/images", false, /\.(png|jpe?g)$/)
);

const PER_PAGE = 12;

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
  const [page, setPage] = useState(1);
  const catBarRef = useRef(null);

  useEffect(() => {
    setCategoryId(currentCategory);
  }, [currentCategory]);

  const changeCat = (id) => {
    setCategoryId(id);
    setPage(1);
    onCategoryChange?.(id);
  };

  const scrollCats = (offset) => {
    if (catBarRef.current) {
      catBarRef.current.scrollBy({ left: offset, behavior: "smooth" });
    }
  };

  const category   = categories.find(c => c.id === categoryId) || categories[0];
  const items      = category.items;
  const totalPages = Math.ceil(items.length / PER_PAGE) || 1;
  const pagedItems = useMemo(
    () => items.slice((page - 1) * PER_PAGE, page * PER_PAGE),
    [items, page]
  );

  const totalQty   = cart.reduce((s, i) => s + (i.qty || 1), 0);
  const totalPrice = cart.reduce((s, i) => s + (i.total || i.price || 0), 0);

  return (
    <div className="menu-screen">
      {/* 헤더 */}
      <header className="app-header">
        <button className="home-btn" onClick={onGoHome}>
          <FaHome size={22}/>
        </button>
        <h1 className="app-title">카페</h1>
      </header>

      {/* 카테고리 + 스크롤 화살표 */}
      <div className="category-nav">
        <button className="cat-nav-btn left" onClick={() => scrollCats(-200)} aria-label="이전 카테고리">
          <FaChevronLeft size={18}/>
        </button>
        <nav ref={catBarRef} className="category-bar">
          {categories.map(c => (
            <div
              key={c.id}
              className={`category-item ${c.id === categoryId ? "active" : ""}`}
              onClick={() => changeCat(c.id)}
            >
              {c.name}
            </div>
          ))}
        </nav>
        <button className="cat-nav-btn right" onClick={() => scrollCats(200)} aria-label="다음 카테고리">
          <FaChevronRight size={18}/>
        </button>
      </div>

      {/* 메뉴 그리드 */}
      <main className="menu-body">
        <div className="menu-grid">
          {pagedItems.map(m => {
            const imgSet = images[m.imageKey] || {};
            const src320 = imgSet["320"];
            const src600 = imgSet["600"];

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
                      sizes={`
                        (max-width: 480px) 100vw,
                        (min-width: 1440px) calc((100vw - 32px*2)/8),
                        160px
                      `}
                      alt={m.name}
                      className="thumb-img"
                      id={m.imageKey}
                      data-name={m.name}
                    />
                  )}
                </div>
                <p className="name">{m.name}</p>
                <p className="price">{m.price.toLocaleString()}원</p>
              </article>
            );
          })}
        </div>

        {/* 페이지 도트 */}
        <div className="pagination">
          {Array.from({ length: totalPages }, (_, i) => i + 1).map(n => (
            <span
              key={n}
              className={`dot ${page === n ? "active" : ""}`}
              onClick={() => setPage(n)}
            />
          ))}
        </div>
      </main>

      {/* 하단 바 */}
      <footer className="checkout-bar">
        <button className="cart-link" onClick={onViewCart}>
          <FaShoppingCart size={18}/> 장바구니 보기
          {totalQty > 0 && <span className="badge">{totalQty}</span>}
        </button>
        <button className="btn-pay" onClick={onCheckout}>
          총 {totalQty}개 {totalPrice.toLocaleString()}원 결제하기
        </button>
      </footer>
    </div>
  );
}
