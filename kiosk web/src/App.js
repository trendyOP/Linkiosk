import React, { useState } from "react";

import PackagingScreen     from "./components/PackagingScreen";
import MenuScreen          from "./components/MenuScreen";
import ItemDetailScreen    from "./components/ItemDetailScreen";
import CartScreen          from "./components/CartScreen";
import PaymentMethodScreen from "./components/PaymentMethodScreen";
import PaymentOptionScreen from "./components/PaymentOptionScreen";
import CardPaymentScreen   from "./components/CardPaymentScreen";
import CompletionScreen    from "./components/CompletionScreen";
import ElderWizard        from "./components/elder/ElderWizard";
import { categories }      from "./data/menuData";      // ← 최초 카테고리 id

function App() {
  /* ─ 상태 ─ */
  const [step, setStep]           = useState(2);
  const [selectedItem, setSel]    = useState(null);
  const [cart, setCart]           = useState([]);
  const [elderMode, setElderMode] = useState(() => new URLSearchParams(window.location.search).has("elder"));
  const [menuCategory, setCat]    = useState(categories[0].id); // ★ 현재 카테고리

  /* 장바구니 담기 */
  const addToCart = (it) => { setCart((p) => [...p, it]); setStep(5); };
  const amount    = cart.reduce((s, i) => s + i.total, 0);
  const resetFlow = () => { setCart([]); setStep(2); };

  /* ─ 화면 결정 ─ */
  let screen;
  switch (step) {
    case 10:
      screen = (
        <ElderWizard cart={cart}
          onAddAndGoCart={(it) => { addToCart(it); }}
          goCart={() => setStep(5)}
          onCancel={() => setStep(3)}
          onGoHome={() => setStep(2)}
        />
      );
      break;
    case 2:
      screen = <PackagingScreen elderMode={elderMode} setElderMode={setElderMode} onNext={() => setStep(elderMode ? 10 : 3)} />;
      break;

    case 3: /* 메뉴 */
      screen = (
        <MenuScreen
          currentCategory={menuCategory}
          onCategoryChange={setCat}             /* ★ 부모에 알려주기 */
          onSelectItem={(it) => { setSel(it); setStep(4); }}
          cart={cart}
          onViewCart={() => setStep(5)}
          onCheckout={() => setStep(6)}
          onSwitchToElder={() => setStep(10)}
          onGoHome={() => setStep(2)}
        />
      );
      break;

    case 4: /* 옵션 모달 */
      screen = (
        <ItemDetailScreen
          item={selectedItem}
          onAdd={addToCart}
          onBack={() => setStep(elderMode ? 10 : 3)}
        />
      );
      break;

    case 5: /* 장바구니 */
      screen = (
        <CartScreen
          cart={cart}
          onBack={() => setStep(elderMode ? 10 : 3)}
          onClear={() => setCart([])}
          onNext={() => setStep(6)}
        />
      );
      break;

    case 6: /* 주문내역 확인 */
      screen = (
        <PaymentMethodScreen
          cart={cart}
          onBack={() => setStep(5)}
          onPay={() => setStep(7)}
        />
      );
      break;

    case 7: /* 결제수단 선택 */
      screen = (
        <PaymentOptionScreen
          amount={amount}
          onCancel={() => setStep(6)}
          onSelect={() => setStep(8)}
        />
      );
      break;

    case 8: /* 카드 결제 + 영수증 */
      screen = (
        <CardPaymentScreen
          amount={amount}
          onComplete={() => setStep(9)}
        />
      );
      break;

    case 9: /* 완료 */
      screen = (
        <CompletionScreen
          orderNumber={Math.floor(Math.random() * 900) + 100}
          onRestart={resetFlow}
        />
      );
      break;

    default:
      screen = <div>잘못된 단계</div>;
  }
  return screen;
}
export default App;
