// src/components/MenuScreen.js
import React, { useMemo, useRef, useState, useEffect } from "react";
import { categories as defaultCategories } from "../data/menuData";

/**
 * 일반 모드 메뉴 화면
 *
 * Props (선택):
 *  - cart, onCheckout, onOpenItem, onAddToCart
 *  - onSwitchToElder: () => void         // 고령자 보기로 전환
 *  - onGoHome: () => void                 // 🏠 첫 화면(포장/매장)으로 이동
 *  - categories: 커스텀 데이터 (없으면 menuData)
 */
export default function MenuScreen({
  cart = [],
  onCheckout,
  onOpenItem,
  onAddToCart,
  onSwitchToElder,
  onGoHome,
  categories = defaultCategories,
}) {

  // useEffect로 마운트 시 fs-normal을 붙이고, 언마운트 시 제거
  useEffect(() => {
    const html = document.documentElement;
    html.classList.add('fs-normal');
    return () => html.classList.remove('fs-normal');
  }, []);

  // 이미지 매핑
  const importAll = (r) =>
    r.keys().reduce((acc, key) => {
      const m = key.match(/\.\/(.+)-(\d+)\.(png|jpe?g)$/);
      if (m) {
        const [, name, size] = m;
        (acc[name] = acc[name] || {})[size] = r(key);
      }
      return acc;
    }, {});
  const images = useMemo(
    () => importAll(require.context("../assets/images", false, /\.(png|jpe?g)$/)),
    []
  );

  // 플랫 아이템
  const allItems = useMemo(() => {
    const arr = [];
    categories.forEach((cat) => (cat.items || []).forEach((it) => arr.push({ ...it, catId: cat.id })));
    return arr;
  }, [categories]);

  // 카테고리
  const [catIndex, setCatIndex] = useState(0);
  const currentCat = categories[catIndex] || { id: null, name: "전체", items: [] };

  // 카테고리 바 스크롤
  const catBarRef = useRef(null);
  const scrollCats = (dir) => {
    const el = catBarRef.current;
    if (!el) return;
    const w = el.clientWidth;
    el.scrollBy({ left: dir * (w * 0.8), behavior: "smooth" });
  };

  // 현재 카테고리 아이템
  const items = useMemo(() => {
    if (!currentCat?.id) return allItems;
    return allItems.filter((it) => it.catId === currentCat.id);
  }, [currentCat, allItems]);

  // 아이템 클릭
  const handleItemClick = (it) => {
    if (typeof onOpenItem === "function") return onOpenItem(it);
    if (typeof onAddToCart === "function") return onAddToCart({ ...it, qty: 1, total: it.price });
  };

  // 장바구니 합계
  const totalQty = cart.reduce((s, i) => s + (i.qty || 1), 0);
  const totalPrice = cart.reduce((s, i) => s + (i.total ?? i.price ?? 0), 0);

  return (
    <div className="menu-screen">
      {/* 헤더 */}
      <header className="app-header">
        <button
          className="home-btn"
          aria-label="처음으로"
          onClick={() => onGoHome?.()}
          title="처음으로"
        >
          🏠
        </button>
        <h1 className="app-title">카페</h1>
        <button
          className="elder-switch"
          onClick={() => onSwitchToElder?.()}
          aria-label="고령자 보기로 전환"
        >
          고령자 보기
        </button>
      </header>

      {/* 카테고리 네비 */}
      <div className="category-nav" role="navigation" aria-label="카테고리">
        <button
          className="cat-nav-btn left"
          onClick={() => setCatIndex((i) => Math.max(0, i - 1))}
          disabled={catIndex <= 0}
          aria-label="이전 카테고리"
        >
          ◀
        </button>

        <div ref={catBarRef} className="category-bar">
          {categories.map((c, idx) => (
            <div
              key={c.id || idx}
              className={"category-item " + (idx === catIndex ? "active" : "")}
              onClick={() => setCatIndex(idx)}
              role="button"
              aria-pressed={idx === catIndex}
            >
              {c.name}
            </div>
          ))}
        </div>

        <button
          className="cat-nav-btn right"
          onClick={() => setCatIndex((i) => Math.min(categories.length - 1, i + 1))}
          disabled={catIndex >= categories.length - 1}
          aria-label="다음 카테고리"
        >
          ▶
        </button>
      </div>

      {/* 메뉴 리스트 */}
      <div className="menu-body">
        <div className="menu-grid">
          {items.map((it) => {
            const src320 = images[it.imageKey]?.["320"];
            const src600 = images[it.imageKey]?.["600"];
            const src = src600 || src320;
            return (
              <div
                key={it.id}
                className="menu-item"
                onClick={() => handleItemClick(it)}
                role="button"
                aria-label={`${it.name} 선택`}
              >
                <div className="thumb">
                  {src ? (
                    <img className="thumb-img" src={src} alt={`${it.name} 이미지`} />
                  ) : (
                    <div className="thumb-img" aria-hidden="true" />
                  )}
                </div>
                <div className="name">{it.name}</div>
                <div className="price">{(it.price || 0).toLocaleString()}원</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 하단 체크아웃 바 */}
      <div className="checkout-bar" aria-live="polite">
        <button className="cart-link" type="button" aria-label="장바구니 보기">
          🛒 장바구니 <span className="badge">{totalQty}</span>
        </button>
        <div style={{ fontWeight: 700 }}>{totalPrice.toLocaleString()}원</div>
        <button className="btn-pay" onClick={() => onCheckout?.()}>결제하기</button>
      </div>
    </div>
  );
}
