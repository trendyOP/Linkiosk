import React, { useState, useRef } from "react";
import { FaChevronLeft, FaChevronRight } from "react-icons/fa";
import { optionTemplates } from "../data/optionData";

// ── 옵션 이미지 600px만 로드 ──
const importOptions600 = (r) =>
  r.keys().reduce((acc, key) => {
    const m = key.match(/\.\/(.+)-600\.(png|jpe?g)$/);
    if (m) acc[m[1]] = r(key);
    return acc;
  }, {});
const optionImages = importOptions600(
  require.context("../assets/option-images", false, /-600\.(png|jpe?g)$/)
);

export default function ItemDetailScreen({ item, onAdd, onBack }) {
  if (!item) return null;
  const { name, price, category, heatable = false } = item;

  let baseOpts = optionTemplates[category] || optionTemplates.default;
  baseOpts = baseOpts.filter((g) => !(g.id === "warm" && !heatable));

  const initSel = {};
  baseOpts.forEach((g) => {
    if (g.required && g.items.length) initSel[g.id] = [g.items[0]];
  });
  const [selected, setSelected] = useState(initSel);
  const [qty, setQty] = useState(1);

  const optionBarRef = useRef(null);
  const scrollOptions = (offset) => {
    if (optionBarRef.current) {
      optionBarRef.current.scrollBy({ left: offset, behavior: "smooth" });
    }
  };

  const toggle = (gid, opt) => {
    setSelected((prev) => {
      const cur = prev[gid] ? [...prev[gid]] : [];
      const { max } = baseOpts.find((g) => g.id === gid);
      const idx = cur.findIndex((o) => o.id === opt.id);
      if (idx > -1) cur.splice(idx, 1);
      else if (max === 1) cur.splice(0, cur.length, opt);
      else if (cur.length < max) cur.push(opt);
      return { ...prev, [gid]: cur };
    });
  };

  const addPrice = Object.values(selected).flat().reduce((s, o) => s + o.price, 0);
  const total = (price + addPrice) * qty;
  const requiredOK = baseOpts
    .filter((g) => g.required)
    .every((g) => selected[g.id]?.length === 1);

  return (
    <div className="modal-overlay">
      <div className="modal">

        {/* 옵션 그룹 */}
        <div className="option-groups">
          {baseOpts.map((grp) => (
            <section key={grp.id} className="option-group">
              <h3>
                {grp.title} {grp.required && <em>(필수)</em>}
              </h3>

              {/* ── 화살표 네비가 필요한 군(`coffee_opts`)만 네비형, 나머진 일반형 ── */}
              {grp.id === "coffee_opts" ? (
                <div className="option-nav">
                  <button
                    className="opt-nav-btn left"
                    onClick={() => scrollOptions(-200)}
                    aria-label="이전 옵션"
                  >
                    <FaChevronLeft size={16} />
                  </button>
                  <div ref={optionBarRef} className="option-items">
                    {grp.items.map((opt) => {
                      const active = selected[grp.id]?.some((o) => o.id === opt.id);
                      const src = optionImages[opt.id];
                      return (
                        <div
                          key={opt.id}
                          className={`option-item ${active ? "active" : ""}`}
                          onClick={() => toggle(grp.id, opt)}
                        >
                          <div className="thumb">
                            {src && (
                              <img
                                src={src}
                                alt={opt.name}
                                className="option-thumb-img"
                              />
                            )}
                          </div>
                          <span>{opt.name}</span>
                          <strong>
                            {opt.price > 0 ? `+${opt.price}원` : "0원"}
                          </strong>
                        </div>
                      );
                    })}
                  </div>
                  <button
                    className="opt-nav-btn right"
                    onClick={() => scrollOptions(200)}
                    aria-label="다음 옵션"
                  >
                    <FaChevronRight size={16} />
                  </button>
                </div>
              ) : (
                <div className="option-items">
                  {grp.items.map((opt) => {
                    const active = selected[grp.id]?.some((o) => o.id === opt.id);
                    const src = optionImages[opt.id];
                    return (
                      <div
                        key={opt.id}
                        className={`option-item ${active ? "active" : ""}`}
                        onClick={() => toggle(grp.id, opt)}
                      >
                        <div className="thumb">
                          {src && (
                            <img
                              src={src}
                              alt={opt.name}
                              className="option-thumb-img"
                            />
                          )}
                        </div>
                        <span>{opt.name}</span>
                        <strong>
                          {opt.price > 0 ? `+${opt.price}원` : "0원"}
                        </strong>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          ))}
        </div>

        {/* 수량 선택 */}
        <div className="qty-box">
          <button onClick={() => qty > 1 && setQty(qty - 1)}>-</button>
          <span>{qty}</span>
          <button onClick={() => setQty(qty + 1)}>+</button>
        </div>

        {/* 푸터 */}
        <div className="modal-foot">
          <button className="btn-cancel" onClick={onBack}>
            취소
          </button>
          <button
            className="btn-primary"
            disabled={!requiredOK}
            onClick={() => onAdd({ ...item, qty, options: selected, total })}
          >
            주문담기 {total.toLocaleString()}원
          </button>
        </div>
      </div>
    </div>
  );
}