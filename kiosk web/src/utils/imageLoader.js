/**
 * 이미지 파일을 require.context로 일괄 로드해서
 * '파일명(확장자 제외)' → 실제 URL 매핑을 만든다.
 * 예) cof_ame-600.jpg → key: 'cof_ame-600'
 */
function importAll(r) {
  const map = {};
  r.keys().forEach((k) => {
    const clean = k.replace(/^\.\//, "").replace(/\.(png|jpe?g|webp|gif|svg)$/i, "");
    map[clean] = r(k);
  });
  return map;
}

// 메뉴/옵션 이미지 폴더를 각각 스캔
const MENU_CTX = require.context("../assets/menu-images", true, /\.(png|jpe?g|webp|gif|svg)$/i);
const OPT_CTX  = require.context("../assets/option-images", true, /\.(png|jpe?g|webp|gif|svg)$/i);

export const MENU_IMAGES = importAll(MENU_CTX);
export const OPTION_IMAGES = importAll(OPT_CTX);

// 안전하게 꺼내는 헬퍼
export const getMenuImage = (key) => MENU_IMAGES[key];
export const getOptionImage = (key) => OPTION_IMAGES[key];
