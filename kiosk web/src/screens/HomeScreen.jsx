import { categories } from "../data/menuData";
import { Link } from "react-router-dom";

export default function HomeScreen() {
  return (
    <main className="home">
      <h1>원하는 메뉴를 선택하세요</h1>
      <div className="grid categories">
        {categories.map((c) => (
          <Link key={c.id} to={`/menu/${c.id}`} className="category-btn">
            {c.label}
          </Link>
        ))}
      </div>
    </main>
  );
}
