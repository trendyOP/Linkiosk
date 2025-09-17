import { useParams } from "react-router-dom";
import { categories, itemsByCategory } from "../data/menuData";
import { getMenuImage } from "../utils/imageLoader";

export default function MenuScreen() {
  const { categoryId } = useParams();
  const category = categories.find((c) => c.id === categoryId);
  const items = itemsByCategory[categoryId] || [];

  return (
    <main className="menu-screen">
      <header className="page-header">
        <h1>메뉴</h1>
        <div className="subtitle">{category?.label}</div>
      </header>

      <div className="grid items">
        {items.map((it) => {
          const src = it.imgKey ? getMenuImage(it.imgKey) : undefined;
          return (
            <button key={it.id} className="item-card">
              {src && <img src={src} alt={`${it.name} 이미지`} />}
              <div className="item-name">{it.name}</div>
              <div className="item-price">{it.price.toLocaleString()}원</div>
            </button>
          );
        })}
      </div>
    </main>
  );
}
