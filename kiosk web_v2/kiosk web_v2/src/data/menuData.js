// src/data/menuData.js
// ─ 한국 프랜차이즈 키오스크용 메뉴 데이터 (이미지키 추가됨) ─

export const categories = [
  /* 1 ─ 커피 */
  {
    id: "coffee",
    name: "커피",
    items: [
      { id: "cof_ame",       name: "아메리카노",            price: 3300, imageKey: "cof_ame" },
      { id: "cof_lat",       name: "카페라떼",              price: 3500, imageKey: "cof_lat" },
      { id: "cof_cap",       name: "카푸치노",              price: 3500, imageKey: "cof_cap" },
      { id: "cof_cm",        name: "카라멜 마끼아또",       price: 3900, imageKey: "cof_cm" },
      { id: "cof_moca",      name: "카페모카",              price: 4000, imageKey: "cof_moca" },
      { id: "cof_cb",        name: "콜드브루 라떼",         price: 4800, imageKey: "cof_cb" },
      { id: "cof_goodhazel", name: "헤이즐넛 (ICE)",        price: 4200, imageKey: "cof_goodhazel" },
      { id: "cof_black",     name: "더 블랙",               price: 4500, imageKey: "cof_black" },
      { id: "bev_vanilla",        name: "바닐라 라떼",             price: 4000, imageKey: "bev_vanilla" },
      { id: "bev_caramelcrunch",  name: "카라멜 크런치",           price: 4500, imageKey: "bev_caramelcrunch" },
      { id: "bev_sheercreamlatte",name: "슈크림라떼",             price: 4500, imageKey: "bev_sheercreamlatte" },
      { id: "cof_bs",            name: "흑당 콜드브루",         price: 4200, imageKey: "cof_bs" }
    ]
  },

  /* 2 ─ 논-커피 음료 */
  {
    id: "noncoffee",
    name: "논-커피 음료",
    items: [
      { id: "nc_choco",  name: "리얼 초코 라떼",   price: 4200, imageKey: "nc_choco" },
      { id: "nc_mint",   name: "민트 초코 라떼",   price: 4400, imageKey: "nc_mint" },
      { id: "nc_chai",   name: "차이 라떼",       price: 4000, imageKey: "nc_chai" },
      { id: "nc_greent", name: "그린티 라떼",     price: 4000, imageKey: "nc_greent" },
      { id: "nc_sweet",  name: "고구마 라떼",     price: 4400, imageKey: "nc_sweet" },
      { id: "nc_black",  name: "흑임자 라떼",     price: 4500, imageKey: "nc_black" }
    ]
  },

  /* 3 ─ 티 · 에이드 */
  {
    id: "tea",
    name: "티 · 에이드",
    items: [
      { id: "tea_cam",            name: "캐모마일 티",           price: 3500, imageKey: "tea_cam" },
      { id: "tea_hib",            name: "히비스커스 티",         price: 3700, imageKey: "tea_hib" },
      { id: "ade_lemon",          name: "레몬 에이드",           price: 4300, imageKey: "ade_lemon" },
      { id: "ade_grape",          name: "자몽 스파클링",         price: 4300, imageKey: "ade_grape" },
      { id: "ade_mus",            name: "샤인머스캣 에이드",     price: 4400, imageKey: "ade_mus" },
      { id: "tea_peach",          name: "복숭아 아이스티",       price: 3200, imageKey: "tea_peach" },
      { id: "ade_pine_lavender",  name: "파인 레몬 라벤더티",   price: 3400, imageKey: "ade_pine_lavender" },
      { id: "ade_peach_lavender", name: "복숭아 레몬 라벤더티", price: 3400, imageKey: "ade_peach_lavender" }
    ]
  },

  /* 4 ─ 디저트 */
  {
    id: "dessert",
    name: "디저트",
    items: [
      { id: "des_ccake", name: "뉴욕 치즈케이크",       price: 5400, heatable: false, imageKey: "des_ccake" },
      { id: "des_tira",  name: "티라미수 컵",           price: 4900, heatable: false, imageKey: "des_tira" },
      { id: "des_mac",   name: "마카롱 4-pcs 세트",     price: 8800, heatable: false, imageKey: "des_mac" },
      { id: "des_brown", name: "리얼 초코 브라우니",     price: 4200, heatable: false, imageKey: "des_brown" },
      { id: "des_yog",   name: "블루베리 요거트 케이크", price: 5200, heatable: false, imageKey: "des_yog" },
      { id: "des_cream", name: "크렘브륄레 컵",         price: 4500, heatable: false, imageKey: "des_cream" }
    ]
  },

  /* 5 ─ 베이커리 · 샌드위치 */
  {
    id: "bakery",
    name: "베이커리 · 샌드위치",
    items: [
      { id: "bak_bagel", name: "플레인 베이글",       price: 2900, heatable: true,  imageKey: "bak_bagel" },
      { id: "bak_crois", name: "버터 크루아상",       price: 3500, heatable: true,  imageKey: "bak_crois" },
      { id: "bak_sand",  name: "햄&치즈 샌드위치",    price: 4700, heatable: false, imageKey: "bak_sand" },
      { id: "bak_croff", name: "플레인 크로플",       price: 3100, heatable: true,  imageKey: "bak_croff" },
      { id: "bak_pretz", name: "갈릭 프레첼",         price: 3300, heatable: false, imageKey: "bak_pretz" },
      { id: "bak_egg",   name: "콘치즈 에그타르트",   price: 3200, heatable: false, imageKey: "bak_egg" }
    ]
  },

  /* 6 ─ 프라푸치노 · 블렌디드 */
  {
    id: "frapp",
    name: "프라푸치노 · 블렌디드",
    items: [
      { id: "frp_java",            name: "자바칩 프라푸치노",    price: 5400, imageKey: "frp_java" },
      { id: "frp_mocha",           name: "모카 프라푸치노",      price: 5200, imageKey: "frp_mocha" },
      { id: "frp_match",           name: "말차 크림 프라푸치노", price: 5400, imageKey: "frp_match" },
      { id: "frp_straw",           name: "딸기 크림 프라푸치노", price: 5200, imageKey: "frp_straw" },
      { id: "frp_caram",           name: "카라멜 프라푸치노",    price: 5200, imageKey: "frp_caram" },
      { id: "frp_cookie",          name: "쿠키&크림 블렌디드",   price: 5400, imageKey: "frp_cookie" },
      { id: "frp_milkshake",       name: "밀크쉐이크 (ICE)",      price: 5000, imageKey: "frp_milkshake" },
      { id: "frp_coffeemilkshake", name: "커피밀크쉐이크 (ICE)",  price: 6000, imageKey: "frp_coffeemilkshake" }
    ]
  },

  /* 7 ─ 스무디 · 주스 */
  {
    id: "smoothie",
    name: "스무디 · 주스",
    items: [
      { id: "sm_berry",     name: "딸기 바나나 스무디",    price: 4900, imageKey: "sm_berry" },
      { id: "sm_mango",     name: "망고 스무디",           price: 4900, imageKey: "sm_mango" },
      { id: "sm_blue",      name: "블루베리 요거트 스무디", price: 5100, imageKey: "sm_blue" },
      { id: "sm_peach",     name: "복숭아 스무디",         price: 4900, imageKey: "sm_peach" },
      { id: "sm_water",     name: "수박 주스",             price: 4200, imageKey: "sm_water" },
      { id: "sm_kiwi",      name: "키위 스무디",          price: 4700, imageKey: "sm_kiwi" },
      { id: "sm_orange",    name: "오렌지 스무디",        price: 4700, imageKey: "sm_orange" },
      { id: "sm_grape",     name: "청포도 스무디",        price: 4700, imageKey: "sm_grape" },
      { id: "sm_grapefruit",name: "자몽 스무디",          price: 4700, imageKey: "sm_grapefruit" },
      { id: "sm_pineapple", name: "파인애플 스무디",      price: 4700, imageKey: "sm_pineapple" }
    ]
  }
];
