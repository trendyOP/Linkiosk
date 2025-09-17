import React, { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMode } from '../context/ModeContext';
import { categories, items } from '../data/menu';

export default function Menu() {
  const { mode } = useMode();
  const navigate = useNavigate();
  const [cat, setCat] = useState('cafe');
  const [cart, setCart] = useState([]);

  const list = useMemo(() => items.filter(i => i.cat === cat), [cat]);

  const add = (it) => {
    setCart(prev => [...prev, it]);
  };

  const goCart = () => {
    navigate('/cart', { state: { cart } });
  };

  const Tab = ({ c }) => (
    <button
      key={c.id}
      className={`tab ${c.id === cat ? 'active' : ''}`}
      onClick={() => setCat(c.id)}
    >
      {c.name}
    </button>
  );

  return (
    <main className="container">
      <div className="tabs-wrap">
        {categories.map(c => <Tab key={c.id} c={c} />)}
      </div>

      <div className={mode === 'elder' ? 'grid elder' : 'grid'}>
        {list.map(it => (
          <button key={it.id} className="card" onClick={() => add(it)}>
            <div className="thumb" aria-hidden>☕</div>
            <div className="title">{it.name}</div>
            <div className="price">{it.price.toLocaleString()}원</div>
          </button>
        ))}
      </div>

      <footer className="cartbar">
        <div>
          선택 <strong>{cart.length}</strong>개 ·{' '}
          <strong>{cart.reduce((s, i) => s + i.price, 0).toLocaleString()}</strong>원
        </div>
        <button className="go-cart" onClick={goCart}>장바구니</button>
      </footer>
    </main>
  );
}
