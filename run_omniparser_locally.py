# c:\Users\chcho\Downloads\OmniParser-master\run_omniparser_locally.py
import os
import time
import cv2 # OpenCV_Python for image display
import numpy as np
import json
from PIL import Image
import torch
import io
import base64

# OmniParser 유틸리티 함수 및 모델 로더 임포트
# util.utils 모듈이 OmniParser-master/util/utils.py 에 있다고 가정합니다.
try:
    from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
except ImportError:
    print("Error: OmniParser's 'util.utils' module not found.")
    print("Please ensure this script is in the 'OmniParser-master' directory and 'util' subdirectory exists.")
    exit()

# --- 모델 로딩 (gradio_demo.py 참조) ---
# 'weights' 디렉토리가 현재 스크립트 위치 기준으로 접근 가능해야 합니다.
YOLO_MODEL_PATH = 'weights/icon_detect/model.pt'
# gradio_demo.py는 BLIP 모델을 기본으로 사용합니다.
CAPTION_MODEL_NAME = "blip"
CAPTION_MODEL_PATH = "Salesforce/blip-image-captioning-base"

print(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
if not os.path.exists(YOLO_MODEL_PATH):
    print(f"Error: YOLO model not found at {YOLO_MODEL_PATH}")
    exit()
yolo_model = get_yolo_model(model_path=YOLO_MODEL_PATH)
print("YOLO model loaded.")

print(f"Loading caption model: {CAPTION_MODEL_NAME} from {CAPTION_MODEL_PATH}")
caption_model_processor = get_caption_model_processor(model_name=CAPTION_MODEL_NAME, model_name_or_path=CAPTION_MODEL_PATH)
print("Caption model and processor loaded.")
# --- 모델 로딩 완료 ---

def run_omniparser_core(image_path: str,
                        box_threshold: float = 0.05,
                        iou_threshold: float = 0.1,
                        use_paddleocr: bool = True, # Gradio 데모 기본값
                        imgsz: int = 480,           # Gradio 데모 기본값
                        output_json_path: str = "omniparser_local_output.json"):
    """
    로컬 이미지에 대해 OmniParser 핵심 처리 로직을 실행합니다.
    """
    try:
        image_input_pil = Image.open(image_path)
    except FileNotFoundError:
        print(f"오류: 이미지 파일을 찾을 수 없습니다: {image_path}")
        return None, None, None
    except Exception as e:
        print(f"오류: 이미지 로딩 중 문제 발생 {image_path}: {e}")
        return None, None, None

    print(f"이미지 처리 중: {image_path}")

    # --- gradio_demo.py의 'process' 함수 로직 모방 ---
    # Bbox 그리기 설정 (gradio_demo.py 참조)
    box_overlay_ratio = image_input_pil.size[0] / 3200.0 # 3200은 참조 너비로 가정
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    # 1. OCR 수행
    print("OCR 수행 중...")
    # check_ocr_box는 내부적으로 EasyOCR 또는 PaddleOCR을 사용할 수 있습니다.
    ocr_bbox_rslt, _ = check_ocr_box(
        image_input_pil,
        display_img=False,
        output_bb_format='xyxy',
        goal_filtering=None,
        easyocr_args={'paragraph': False, 'text_threshold': 0.9}, # gradio_demo.py 설정값
        use_paddleocr=use_paddleocr
    )
    text_ocr, ocr_bbox = ocr_bbox_rslt
    print(f"OCR 결과: {len(text_ocr)}개의 텍스트 요소 발견.")

    # 2. 객체 탐지 및 레이블링 (get_som_labeled_img 호출)
    print("객체 탐지 및 레이블링 수행 중...")
    # get_som_labeled_img는 OmniParser의 핵심 유틸리티 함수입니다.
    # 전역으로 로드된 yolo_model과 caption_model_processor를 사용합니다.
    labeled_img_base64, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_input_pil,
        yolo_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True, # gradio_demo.py 설정값
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text_ocr,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
    )
    print("객체 탐지 및 레이블링 완료.")

    # 결과 이미지 디코딩 (Base64 -> PIL Image)
    processed_image_pil = None
    try:
        if labeled_img_base64:
            processed_image_pil = Image.open(io.BytesIO(base64.b64decode(labeled_img_base64)))
    except Exception as e:
        print(f"오류: Base64 이미지 디코딩 중 문제 발생: {e}")

    # 파싱된 콘텐츠 포맷팅 (Gradio 데모와 유사하게)
    formatted_parsed_content_summary = '\n'.join([f'element {i}: {str(v)}' for i, v in enumerate(parsed_content_list)])

    # 파싱된 콘텐츠를 JSON으로 저장 (구조화된 리스트 자체를 저장)
    try:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_content_list, f, indent=4, ensure_ascii=False)
        print(f"✔ 파싱된 콘텐츠 저장 완료: {output_json_path}")
    except Exception as e:
        print(f"오류: JSON 출력 저장 중 문제 발생: {e}")

    return processed_image_pil, formatted_parsed_content_summary, parsed_content_list

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 테스트용 이미지 경로 설정
    # action.py에서 사용된 R2.html로 스크린샷을 생성하거나, 기존 이미지 파일 경로를 사용합니다.
    html_file_for_screenshot = "R1.html"
    generated_screenshot_path = os.path.join(current_script_dir, "screen_omniparsertest.png")
    image_to_process = ""

    # R1.html 파일 존재 여부 확인
    r2_html_path = os.path.join(current_script_dir, html_file_for_screenshot)
    if os.path.exists(r2_html_path):
        print("Selenium을 사용하여 테스트 스크린샷 생성 시도 중...")
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            
            # ChromeDriver 경로 확인 (action.py와 동일한 위치 또는 Linkiosk 폴더 내)
            chromedriver_exe_name = 'chromedriver.exe'
            chromedriver_path_options = [
                os.path.join(current_script_dir, chromedriver_exe_name),
                os.path.join(current_script_dir, '..', 'Linkiosk', chromedriver_exe_name) # GitHub Linkiosk 구조 고려
            ]
            chromedriver_final_path = None
            for path_option in chromedriver_path_options:
                if os.path.exists(path_option):
                    chromedriver_final_path = os.path.abspath(path_option)
                    break
            
            if chromedriver_final_path:
                print(f"ChromeDriver 사용: {chromedriver_final_path}")
                service = Service(executable_path=chromedriver_final_path)
                options = webdriver.ChromeOptions()
                options.add_argument("--start-maximized")
                # options.add_argument("--headless") # 백그라운드 실행 원할 시 주석 해제
                # options.add_argument("--disable-gpu") # Headless 시 종종 필요
                driver = webdriver.Chrome(service=service, options=options)
                
                driver.get(f"file:///{os.path.abspath(r2_html_path)}")
                time.sleep(2) # 페이지 로딩 대기
                
                driver.save_screenshot(generated_screenshot_path)
                print(f"스크린샷 저장 완료: {generated_screenshot_path}")
                driver.quit()
                image_to_process = generated_screenshot_path
            else:
                print(f"경고: ChromeDriver를 찾을 수 없습니다. ({chromedriver_path_options[0]} 또는 {chromedriver_path_options[1]})")

        except ImportError:
            print("경고: Selenium이 설치되어 있지 않습니다. (`pip install selenium`)")
        except Exception as e:
            print(f"경고: 스크린샷 생성 중 오류 발생: {e}")
    else:
        print(f"경고: {html_file_for_screenshot} 파일을 찾을 수 없습니다. 스크린샷을 생성할 수 없습니다.")

    if not image_to_process or not os.path.exists(image_to_process):
        print("\n스크린샷을 생성하지 못했거나, 생성된 이미지를 찾을 수 없습니다.")
        fallback_image = os.path.join(current_script_dir, "screen2.png") # action.py의 결과물
        if os.path.exists(fallback_image):
            print(f"{fallback_image}를 사용합니다.")
            image_to_process = fallback_image
        else:
            image_to_process = input(f"처리할 이미지 파일의 전체 경로를 입력하세요: ")
            if not os.path.exists(image_to_process):
                print(f"이미지 파일을 찾을 수 없습니다: {image_to_process}. 스크립트를 종료합니다.")
                exit()
    
    # OmniParser 실행 파라미터 (gradio_demo.py의 기본값 사용)
    param_box_threshold = 0.05
    param_iou_threshold = 0.1
    param_use_paddleocr = True  # PaddleOCR 사용 여부 (True 또는 False)
    param_imgsz = 480           # 아이콘 탐지 이미지 크기

    print(f"\n--- OmniParser 로컬 실행 시작 ---")
    print(f"이미지: {image_to_process}")
    print(f"Box Threshold: {param_box_threshold}")
    print(f"IOU Threshold: {param_iou_threshold}")
    print(f"PaddleOCR 사용: {param_use_paddleocr}")
    print(f"Icon Detect Image Size (imgsz): {param_imgsz}")
    print(f"---------------------------------\n")

    output_json_file = os.path.join(current_script_dir, "omniparser_local_output.json")
    processed_pil_image, parsed_summary, raw_parsed_list = run_omniparser_core(
        image_path=image_to_process,
        box_threshold=param_box_threshold,
        iou_threshold=param_iou_threshold,
        use_paddleocr=param_use_paddleocr,
        imgsz=param_imgsz,
        output_json_path=output_json_file
    )

    if processed_pil_image:
        print("\n--- 파싱된 콘텐츠 요약 ---")
        print(parsed_summary)
        # print("\n--- 원본 파싱 데이터 리스트 (일부) ---")
        # if raw_parsed_list:
        #     for i, item in enumerate(raw_parsed_list[:3]): # 처음 3개 항목만 출력
        #         print(f"Item {i}: {item}")


        # 처리된 이미지 화면에 표시 (OpenCV 사용)
        try:
            img_cv2_display = cv2.cvtColor(np.array(processed_pil_image), cv2.COLOR_RGB2BGR)
            cv2.imshow("OmniParser Processed Image (Local)", img_cv2_display)
            print("\n처리된 이미지를 표시합니다. 창을 닫으려면 아무 키나 누르세요.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"오류: OpenCV로 이미지 표시 중 문제 발생: {e}")
            print("처리된 이미지는 파일로 저장되지 않지만, JSON 결과는 생성되었을 수 있습니다.")
    else:
        print("OmniParser 처리 실패.")

    print("\nOmniParser 로컬 실행 완료.")

