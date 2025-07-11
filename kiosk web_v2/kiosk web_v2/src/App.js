import React, { useState, useEffect } from "react";

import PackagingScreen     from "./components/PackagingScreen";
import MenuScreen          from "./components/MenuScreen";
import ItemDetailScreen    from "./components/ItemDetailScreen";
import CartScreen          from "./components/CartScreen";
import PaymentMethodScreen from "./components/PaymentMethodScreen";
import PaymentOptionScreen from "./components/PaymentOptionScreen";
import CardPaymentScreen   from "./components/CardPaymentScreen";
import CompletionScreen    from "./components/CompletionScreen";
import { categories }      from "./data/menuData";      // ← 최초 카테고리 id

function App() {
  /* ─ 상태 ─ */
  const [step, setStep]           = useState(2);
  const [selectedItem, setSel]    = useState(null);
  const [cart, setCart]           = useState([]);
  const [menuCategory, setCat]    = useState(categories[0].id); // ★ 현재 카테고리

  /* 장바구니 상태를 PyQt5에 알리는 함수 */
  const notifyCartChange = (newCart) => {
    if (window.pyqtBridge) {
      const cartState = {
        itemCount: newCart.length,
        totalItems: newCart.reduce((sum, item) => sum + item.qty, 0),
        totalPrice: newCart.reduce((sum, item) => sum + item.total, 0),
        items: newCart.map(item => ({
          name: item.name,
          qty: item.qty,
          price: item.total,
          options: item.options
        })),
        step: step
      };
      window.pyqtBridge.notifyCartChange(cartState);
    }
  };

  /* 장바구니 담기 */
  const addToCart = (it) => { 
    const newCart = [...cart, it];
    setCart(newCart); 
    setStep(5);
    notifyCartChange(newCart);
  };

  /* 장바구니에서 아이템 제거 */
  const removeFromCart = (index) => {
    const newCart = cart.filter((_, i) => i !== index);
    setCart(newCart);
    setStep(3);
    notifyCartChange(newCart);
  };

  /* 장바구니 비우기 */
  const clearCart = () => {
    setCart([]);
    notifyCartChange([]);
  };

  const amount    = cart.reduce((s, i) => s + i.total, 0);
  const resetFlow = () => { 
    setCart([]); 
    setStep(2);
    notifyCartChange([]);
  };

  /* 장바구니 상태 변경 시 PyQt5에 알림 */
  useEffect(() => {
    notifyCartChange(cart);
  }, [cart, step]);

  /* ─ 화면 결정 ─ */
  let screen;
  switch (step) {
    case 2:
      screen = <PackagingScreen onNext={() => setStep(3)} />;
      break;

    case 3: /* 메뉴 */
      screen = (
        <MenuScreen
          currentCategory={menuCategory}
          onCategoryChange={setCat}             /* ★ 부모에 알려주기 */
          onGoHome={() => setStep(2)}
          onSelectItem={(it) => { setSel(it); setStep(4); }}
          cart={cart}
          onViewCart={() => setStep(5)}
          onCheckout={() => setStep(6)}
        />
      );
      break;

    case 4: /* 옵션 모달 */
      screen = (
        <ItemDetailScreen
          item={selectedItem}
          onAdd={addToCart}
          onBack={() => setStep(3)}
        />
      );
      break;

    case 5: /* 장바구니 */
      screen = (
        <CartScreen
          cart={cart}
          onBack={() => setStep(3)}
          onClear={clearCart}
          onRemove={removeFromCart}
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
