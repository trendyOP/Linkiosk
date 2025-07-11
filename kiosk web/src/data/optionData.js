// src/data/optionData.js
// ─ 카테고리별 옵션 템플릿 ─

export const optionTemplates = {
  /* coffee ─ 아메리카노·라떼 등 */
  coffee: [
    {
      id: "size",
      title: "사이즈 선택",
      required: true,
      max: 1,
      items: [
        { id: "regular",   name: "기본사이즈",                     price: 0 },
        { id: "upgrade1l", name: "사이즈업(1리터) - ICE 전용",      price: 1400 }
      ]
    },
    {
      id: "temp",
      title: "온도",
      required: true,
      max: 1,
      items: [
        { id: "ice", name: "ICE", price: 0 },
        { id: "hot", name: "HOT", price: 0 }
      ]
    },
    {
      id: "coffee_opts",
      title: "커피 옵션 (최대 8개)",
      required: false,
      max: 8,
      items: [
        { id: "shot1",      name: "1샷 추가",          price: 500 },
        { id: "shot2",      name: "2샷 추가",          price: 1000 },
        { id: "weak",       name: "연하게(1샷)",       price: 0 },
        { id: "decaf",      name: "디카페인으로 변경",  price: 500 },
        { id: "less_water", name: "물 양 적게",        price: 0 },
        { id: "more_ice",   name: "얼음 많이",         price: 0 },
        { id: "less_sugar", name: "덜 달게",           price: 0 },
        { id: "syrup",      name: "설탕 시럽 추가",    price: 0 }
      ]
    },
    {
      id: "tumbler",
      title: "텀블러 사용 (최대 1개)",
      required: false,
      max: 1,
      items: [
        { id: "tumbler", name: "텀블러 사용", price: 0 }
      ]
    }
  ],

  /* noncoffee ─ 논-커피 음료 */
  noncoffee: [
    {
      id: "size",
      title: "사이즈 선택",
      required: true,
      max: 1,
      items: [
        { id: "regular", name: "기본사이즈",            price: 0 },
        { id: "large",   name: "사이즈업(+600원)",      price: 600 }
      ]
    },
    {
      id: "whip",
      title: "휘핑 크림",
      required: false,
      max: 1,
      items: [
        { id: "whip_add", name: "휘핑 추가", price: 500 }
      ]
    }
  ],

  /* tea · ade ─ 티·에이드 */
  tea: [
    {
      id: "sweet",
      title: "당도",
      required: true,
      max: 1,
      items: [
        { id: "reg",  name: "기본",       price: 0 },
        { id: "half", name: "Half Sugar", price: 0 },
        { id: "zero", name: "Zero",       price: 0 }
      ]
    },
    {
      id: "ice",
      title: "얼음량",
      required: false,
      max: 1,
      items: [
        { id: "ice_less", name: "Less Ice", price: 0 },
        { id: "ice_none", name: "No Ice",   price: 0 }
      ]
    }
  ],

  /* dessert ─ 디저트 (옵션 없음) */
  dessert: [],

  /* bakery ─ 베이커리·샌드위치 */
  bakery: [
    {
      id: "warm",
      title: "데우기",
      required: false,
      max: 1,
      items: [
        { id: "heat20", name: "전자레인지 20초", price: 0 }
      ]
    }
  ],

  /* frapp ─ 프라푸치노·블렌디드 */
  frapp: [
    {
      id: "size",
      title: "사이즈 선택",
      required: true,
      max: 1,
      items: [
        { id: "regular", name: "기본사이즈",            price: 0 },
        { id: "large",   name: "사이즈업(+600원)",      price: 600 }
      ]
    },
    {
      id: "base",
      title: "우유/두유 선택",
      required: false,
      max: 1,
      items: [
        { id: "milk", name: "우유", price: 0 },
        { id: "soy",  name: "두유", price: 400 },
        { id: "oat",  name: "오트", price: 600 }
      ]
    }
  ],

  /* smoothie ─ 스무디·주스 */
  smoothie: [
    {
      id: "size",
      title: "사이즈 선택",
      required: true,
      max: 1,
      items: [
        { id: "regular", name: "기본사이즈",            price: 0 },
        { id: "large",   name: "사이즈업(+600원)",      price: 600 }
      ]
    }
  ],

  /* default ─ 기타 카테고리(옵션 없음) */
  default: []
};
