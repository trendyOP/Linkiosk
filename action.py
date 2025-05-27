import os
import time
import cv2
import numpy as np
import json
import pyautogui
import easyocr             # ← OCR 라이브러리
from PIL import ImageFont, ImageDraw, Image  # ← 한글 표시용
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from ultralytics import YOLO

# 한글 텍스트 이미지 위에 표시 (cv2.putText 대체용)
def draw_hangul_text(img_cv2, text, position, font_path="C:/Windows/Fonts/malgun.ttf", font_size=20, color=(255,0,0)):
    img_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 0. OCR 리더 초기화 (한 번만)
reader = easyocr.Reader(['ko', 'en'], gpu=True)

# 1. HTML 띄우기
service = Service('./chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=service, options=options)

html_path = os.path.abspath("R2.html")
driver.get(f"file:///{html_path}")
time.sleep(2)

# 2. 스크린샷 저장
screenshot_path = "screen2.png"
with open(screenshot_path, "wb") as f:
    f.write(driver.get_screenshot_as_png())
    
# 3. 이미지 불러오기
img = cv2.imread(screenshot_path)

# 4. YOLO 모델 로드 및 추론
model = YOLO("weights/icon_detect/model.pt")
results = model(img)
boxes = results[0].boxes.xyxy.cpu().numpy()  # shape [N,4]

# --- OCR 포함 JSON 저장 로직 ---
boxes_int = [list(map(int, b)) for b in boxes]
data = []
for x1, y1, x2, y2 in boxes_int:
    # 중심점 계산
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # ① ROI 자르기
    roi = img[y1:y2, x1:x2]
    # ② OCR 수행 (detail=0: 텍스트만 리스트로 반환)
    texts = reader.readtext(roi, detail=0)
    # ③ 결과 문자열로 병합
    text = " ".join(texts).strip()

    # ✅ 콘솔 한글 깨짐 방지 출력
    try:
        print(text.encode('utf-8').decode('utf-8'))
    except UnicodeEncodeError:
        print(text)

    data.append({
        "box":   [x1, y1, x2, y2],
        "center":[cx, cy],
        "text":  text
    })

# ④ JSON으로 저장
json_path = r"C:\Users\chcho\Downloads\OmniParser-master\boxes2.json"
with open(json_path, "w", encoding="utf-8") as jf:
    json.dump(data, jf, ensure_ascii=False, indent=4)
print(f"✅ {len(data)}개 박스+텍스트 저장 → {json_path}")
# --- OCR 포함 JSON 저장 끝 ---

# 5. 파싱된 모든 박스 시각화 + "부대찌개" 클릭
if len(data) > 0:
    # 시각화
    vis_img = img.copy()
    for entry in data:
        x1, y1, x2, y2 = entry["box"]
        cx_b, cy_b = entry["center"]
        text = entry["text"] 
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(vis_img, (cx_b, cy_b), 5, (0,0,255), -1)
         # 박스 안쪽 상단에 텍스트 삽입 (10px 여백)
        text_position = (x1 + 5, y1 + 5)
        vis_img = draw_hangul_text(vis_img, text, text_position, font_size=20)

    # 시각화 출력
    cv2.imshow("Detected Boxes + OCR", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # "부대찌개"가 포함된 박스 찾기
    target = None
    for entry in data:
        if "아이스크림" in entry["text"]:
            target = entry
            break

    if target:
        x1, y1, x2, y2 = target["box"]
        cx, cy = target["center"]

        # 브라우저 위치 오프셋 보정
        pos = driver.get_window_position()
        chrome_offset_y = 120  # 필요에 따라 조정
        click_x = pos['x'] + cx
        click_y = pos['y'] + cy + chrome_offset_y

        print(f"'부대찌개' 클릭 위치: ({click_x}, {click_y})")
        pyautogui.moveTo(click_x, click_y)
        pyautogui.click()
    else:
        print("❌ '부대찌개' 메뉴를 찾을 수 없습니다.")
else:
    print("No UI elements detected.")
