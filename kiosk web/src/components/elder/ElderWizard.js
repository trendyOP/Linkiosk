// src/components/elder/ElderWizard.js
import React, { useMemo, useState, useEffect } from "react";
import { categories } from "../../data/menuData";

export default function ElderWizard({ onCancel, onAddAndGoCart, goCart, cart = [], onGoHome }) {
  useEffect(() => {
    document.documentElement.classList.remove('fs-normal');
  }, []);

  const [step, setStep]   = useState(1);                         // 1: 따/차, 2: 메뉴, 3: 담기
  const [thermo, setThermo] = useState(null);                    // COF_HOT | COF_COLD | NON_HOT | NON_COLD | DESSERT
  const [pick, setPick]   = useState(null);                      // 선택한 메뉴명

  // 이름 -> 항목 매핑
  const itemByName = useMemo(() => {
    const m = {};
    categories.forEach(cat => (cat.items || []).forEach(it => { m[it.name] = { ...it, catId: cat.id }; }));
    return m;
  }, []);

  // 이미지 로더 (assets/images 의 320/600 썸네일)
  const importAll = (r) =>
    r.keys().reduce((acc, key) => {
      const match = key.match(/\.\/(.+)-(\d+)\.(png|jpe?g)$/);
      if (match) {
        const [, name, size] = match;
        (acc[name] = acc[name] || {})[size] = r(key);
      }
      return acc;
    }, {});
  const images = importAll(require.context("../../assets/images", false, /\.(png|jpe?g)$/));

  // 전체 항목
  const allItems = useMemo(() => {
    const arr = [];
    categories.forEach(cat => (cat.items || []).forEach(it => arr.push({ ...it, catId: cat.id })));
    return arr;
  }, []);

  // ── 분류 헬퍼(중복 선언 금지) ──
  const isCoffee = (id) => id === "coffee";
  const isBeverageNonCoffee = (id) => ["noncoffee", "tea", "frapp", "smoothie"].includes(id);
  const isDessert = (id) => ["dessert", "bakery"].includes(id);

  // 이름에 (ICE) 표기가 있는지 (예: "아메리카노 (ICE)")
  const hasParenICE = (name) => /\(\s*ICE\s*\)/i.test(String(name || ""));

  // 차가운 키워드
  const isColdKeyword = (name) =>
    /(ICE|아이스|콜드|에이드|프라푸치노|스무디|주스|스파클링|쉐이크|블렌디드|아이스티)/i
      .test(String(name || ""));

  // 2단계 목록 규칙
  const list = useMemo(() => {
    switch (thermo) {
      case "COF_HOT": {
        // 뜨거운 커피 = 커피 중 (ICE) 없는 항목에서 ["콜드브루 라떼","흑당 콜드브루"] 제외
        const EXCLUDE = new Set(["콜드브루 라떼", "흑당 콜드브루"]);
        return allItems.filter(it => isCoffee(it.catId) && !hasParenICE(it.name) && !EXCLUDE.has(it.name));
      }
      case "COF_COLD": {
        // 차가운 커피 = 커피 전체(= 뜨거운 커피에 있는 모든 음료도 포함)
        return allItems.filter(it => isCoffee(it.catId));
      }
      case "NON_HOT":
        // 따뜻한 음료 = 논-커피 중 '차가운' 키워드 없는 항목만
        return allItems.filter(it => isBeverageNonCoffee(it.catId) && !isColdKeyword(it.name));
      case "NON_COLD":
        // 차가운 음료 = 논-커피 전체(따뜻한 항목도 포함)
        return allItems.filter(it => isBeverageNonCoffee(it.catId));
      case "DESSERT":
        // 간식(디저트/베이커리)
        return allItems.filter(it => isDessert(it.catId));
      default:
        return [];
    }
  }, [thermo, allItems]);

  const selectedItem = pick ? itemByName[pick] : null;

  // 담기
  const addPicked = () => {
    if (!selectedItem) return;
    const it = selectedItem;
    // thermo 정보는 기록만 (가격은 원본 사용)
    onAddAndGoCart && onAddAndGoCart({ ...it, qty: 1, options: { thermo }, total: it.price });
  };

  // 접근성 토글
  const toggleBig  = () => document.body.classList.toggle("elder-big");
  const toggleHC   = () => document.body.classList.toggle("elder-hc");

  return (
    <div className="elder-wizard">
      {/* 헤더 */}
      <header className="elder-head">
        <button className="home-btn" onClick={() => onGoHome?.()} aria-label="처음으로">🏠</button>
        <h1 className="elder-title">카페</h1>
        <div className="elder-a11y">
          <button className="a11y-btn" onClick={toggleBig}>큰 글자</button>
          <button className="a11y-btn" onClick={toggleHC}>고대비</button>
          <button className="a11y-btn" onClick={onCancel}>일반보기</button>
        </div>
      </header>

      {/* 단계 네비 */}
      <nav className="elder-steps" aria-label="진행 단계">
        <div className={`step ${step===1?'on':''}`} aria-current={step===1}>1. 따뜻/차가운</div>
        <div className={`step ${step===2?'on':''}`} aria-current={step===2}>2. 메뉴</div>
        <div className={`step ${step===3?'on':''}`} aria-current={step===3}>3. 담기</div>
      </nav>

      {/* 본문 */}
      <main className="elder-main" aria-live="polite">
        {/* 1단계: 두 열, 버튼 안에 '선택' */}
        {step===1 && (
          <section className="elder-grid two-col">
            {[{label:"따뜻한 커피", value:"COF_HOT"},
              {label:"차가운 커피", value:"COF_COLD"},
              {label:"따뜻한 음료", value:"NON_HOT"},
              {label:"차가운 음료", value:"NON_COLD"},
              {label:"간식", value:"DESSERT"}].map(opt => (
              <article key={opt.value} className="elder-choice">
                <div className="left"><div className="label">{opt.label}</div></div>
                <div className="right">
                  <button className="btn-xxl primary" onClick={() => { setThermo(opt.value); setStep(2); }}>선택</button>
                </div>
              </article>
            ))}
          </section>
        )}

        {/* 2단계: 메뉴 4열 카드 + 담기 누르면 3단계로 이동 */}
        {step===2 && (
          <section className="elder-grid elder-grid--menu">
            {list.map(it => {
              const src320 = images[it.imageKey]?.["320"];
              const src600 = images[it.imageKey]?.["600"];
              const src = src600 || src320;
              return (
                <article key={it.id} className="elder-card">
                  <figure className="card-media">
                    {src
                      ? <img className="card-thumb" src={src} alt={`${it.name} 이미지`} />
                      : <div className="card-thumb card-thumb--ph" aria-hidden="true" />
                    }
                  </figure>

                  <div className="card-text">
                    <div className="card-title">{it.name}</div>
                    <div className="card-price">{(it.price || 0).toLocaleString()}원</div>
                  </div>

                  <button
                    className="btn-xl primary card-cta"
                    onClick={() => { setPick(it.name); setStep(3); }}
                  >
                    담기
                  </button>
                </article>
              );
            })}
          </section>
        )}

        {/* 3단계: 취소 + 담기(장바구니에 넣기) */}
        {step===3 && (
          <section className="elder-grid">
            <article className="elder-choice picked">
              <div className="left">
                <div className="label">{pick || "선택 없음"}</div>
                <div className="price">{(selectedItem?.price || 0).toLocaleString()}원</div>
              </div>
              <div className="right" style={{display:"flex",gap:"8px"}}>
                <button className="btn-lg ghost" onClick={() => { setPick(null); setStep(2); }}>취소</button>
                <button className="btn-lg primary" onClick={() => { addPicked(); }}>담기</button>
              </div>
            </article>
          </section>
        )}
      </main>

      {/* 하단 고정 바: 장바구니 합계 기준 */}
      <footer className="elder-sticky">
        <div className="sum">
          선택 {cart.reduce((s,i)=>s+(i.qty||1),0)}개 · {cart.reduce((s,i)=>s+(i.total||i.price||0),0).toLocaleString()}원
        </div>
        <div className="actions">
          <button className="btn-lg" onClick={() => setStep(Math.max(1, step-1))}>뒤로가기</button>
          <button className="btn-xxl primary" onClick={() => { if (selectedItem) addPicked(); goCart && goCart(); }}>
            장바구니
          </button>
        </div>
      </footer>
    </div>
  );
}
