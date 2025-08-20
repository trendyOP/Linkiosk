import cv2
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
from PIL import Image
import base64
from io import BytesIO
# import easyocr  # 지연 import로 변경
import pytesseract
# 지연 import로 변경 (PaddleOCR을 먼저 초기화하기 위해)
# from ultralytics import YOLO
# import supervision as sv
# import torch
# from transformers import BlipProcessor, BlipForConditionalGeneration

class ButtonDetector:
    def __init__(self, model_path: str = "weights/icon_detect/model.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        # 지연 import
        from ultralytics import YOLO
        import supervision as sv
        self.model = YOLO(model_path)
        self.box_annotator = sv.BoxAnnotator()
        
    def detect(self, image: np.ndarray) -> List[Dict]:
        # 지연 import
        import supervision as sv
        results = self.model(image)[0]
        detections = sv.Detections.from_yolov8(results)
        
        buttons = []
        for box, confidence, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
            x1, y1, x2, y2 = map(int, box)
            buttons.append({
                'name': f'button_{class_id}',
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence)
            })
        return buttons

class StampDetector:
    def __init__(self):
        self.stamp_color_lower = np.array([0, 100, 100])  # HSV 색상 범위 (빨간색)
        self.stamp_color_upper = np.array([10, 255, 255])
        
    def detect(self, image: np.ndarray) -> Tuple[int, int]:
        # HSV 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 색상 마스크 생성
        mask = cv2.inRange(hsv, self.stamp_color_lower, self.stamp_color_upper)
        
        # 노이즈 제거
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return (0, 0)
            
        # 가장 큰 윤곽선 선택
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        if M["m00"] == 0:
            return (0, 0)
            
        # 중심점 계산
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return (cx, cy)

# 전역 인스턴스 (지연 생성)
_button_detector = None
_stamp_detector = None
_paddle_ocr = None  # PaddleOCR 싱글톤
_easyocr_reader = None  # EasyOCR 싱글톤

def load_image(image_path: str) -> np.ndarray:
    """이미지 로드"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    return image

def detect_buttons(image: np.ndarray) -> List[Dict]:
    """버튼 감지 (YOLO 모델 사용)"""
    global _button_detector
    if _button_detector is None:
        _button_detector = ButtonDetector()
    return _button_detector.detect(image)

def get_stamp_position(image: np.ndarray) -> Tuple[int, int]:
    """도장 위치 감지"""
    global _stamp_detector
    if _stamp_detector is None:
        _stamp_detector = StampDetector()
    return _stamp_detector.detect(image)

def create_paddle_ocr(lang='korean', use_angle_cls=False, show_log=False, **kwargs):
    """
    GPU 사용 가능 여부를 체크하여 자동으로 GPU/CPU를 선택하는 PaddleOCR 팩토리 함수
    
    Args:
        lang: 언어 설정 (기본값: 'korean')
        use_angle_cls: 각도 분류 사용 여부 (기본값: False)
        show_log: 로그 표시 여부 (기본값: False)
        **kwargs: 추가 PaddleOCR 설정
        
    Returns:
        PaddleOCR 인스턴스
    """
    from paddleocr import PaddleOCR
    
    # GPU 사용 가능 여부 체크
    gpu_available = False
    try:
        import paddle
        gpu_available = paddle.device.is_compiled_with_cuda()
        print(f"[PaddleOCR] CUDA 컴파일 여부: {gpu_available}")
    except Exception as e:
        print(f"[PaddleOCR] GPU 체크 실패: {e}")
        gpu_available = False
    
    # GPU 사용 가능하면 GPU로, 아니면 CPU로 초기화
    try:
        if gpu_available:
            print("[PaddleOCR] GPU 모드로 초기화 중...")
            # PaddleOCR 2.x 스타일 시도
            try:
                ocr = PaddleOCR(
                    lang=lang,
                    use_angle_cls=use_angle_cls,
                    show_log=show_log,
                    use_gpu=True,
                    **kwargs
                )
                print("[PaddleOCR] GPU 모드 초기화 성공 (2.x 스타일)")
                return ocr
            except TypeError:
                # PaddleOCR 3.x 스타일 시도
                ocr = PaddleOCR(
                    lang=lang,
                    use_angle_cls=use_angle_cls,
                    show_log=show_log,
                    device='gpu',
                    gpu_id=0,
                    **kwargs
                )
                print("[PaddleOCR] GPU 모드 초기화 성공 (3.x 스타일)")
                return ocr
        else:
            print("[PaddleOCR] CPU 모드로 초기화 중...")
            # CPU 모드 (2.x 스타일)
            try:
                ocr = PaddleOCR(
                    lang=lang,
                    use_angle_cls=use_angle_cls,
                    show_log=show_log,
                    use_gpu=False,
                    **kwargs
                )
                print("[PaddleOCR] CPU 모드 초기화 성공 (2.x 스타일)")
                return ocr
            except TypeError:
                # CPU 모드 (3.x 스타일)
                ocr = PaddleOCR(
                    lang=lang,
                    use_angle_cls=use_angle_cls,
                    show_log=show_log,
                    device='cpu',
                    **kwargs
                )
                print("[PaddleOCR] CPU 모드 초기화 성공 (3.x 스타일)")
                return ocr
                
    except Exception as e:
        print(f"[PaddleOCR] 초기화 실패, CPU로 폴백: {e}")
        # 최종 CPU 폴백
        try:
            ocr = PaddleOCR(
                lang=lang,
                use_angle_cls=use_angle_cls,
                show_log=show_log,
                use_gpu=False,
                **kwargs
            )
            print("[PaddleOCR] CPU 폴백 초기화 성공")
            return ocr
        except TypeError:
            ocr = PaddleOCR(
                lang=lang,
                use_angle_cls=use_angle_cls,
                show_log=show_log,
                device='cpu',
                **kwargs
            )
            print("[PaddleOCR] CPU 폴백 초기화 성공 (3.x 스타일)")
            return ocr

def get_paddle_ocr():
    """PaddleOCR 싱글톤 인스턴스 반환 (가장 먼저 초기화)"""
    global _paddle_ocr
    if _paddle_ocr is None:
        # torch가 로드되기 전에 PaddleOCR 초기화
        _paddle_ocr = create_paddle_ocr(
            lang='korean',               # 한국어로 변경 (매장식사 인식 향상)
            use_angle_cls=False,         # 각도 분류 비활성화 
            show_log=False,
            det_limit_side_len=1920,     # 화면 캡처에 맞춤
            drop_score=0.35,             # 임계값을 0.35로 낮춤 (한국어 텍스트 구제)
            use_gpu=False,               # GPU 비활성화 
            use_dilation=True,           # 정확도 향상 
            det_db_score_mode='slow',    # 정확도 향상 
            max_batch_size=1024,         # 배치 크기 
            rec_batch_num=1024           # 배치 크기
        )
    return _paddle_ocr

def _get_easyocr_reader(langs=('ko','en'), gpu=True, models_dir='./easyocr_models'):
    """EasyOCR 싱글톤 인스턴스 반환 (지연 import)"""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr  # 지연 import (torch가 여기서 로드됨)
        _easyocr_reader = easyocr.Reader(list(langs), gpu=gpu, model_storage_directory=models_dir)
    return _easyocr_reader

def visualize_detections(image: np.ndarray, 
                        buttons: List[Dict], 
                        stamp_pos: Tuple[int, int]) -> np.ndarray:
    """감지 결과 시각화"""
    vis_image = image.copy()
    
    # 버튼 박스 그리기
    for button in buttons:
        x1, y1, x2, y2 = button['bbox']
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_image, button['name'], (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # 도장 위치 표시
    cv2.circle(vis_image, stamp_pos, 5, (0, 0, 255), -1)
    
    return vis_image 

# 공통 유틸 함수들
def _crop_roi(img_np: np.ndarray, roi: Tuple[float,float,float,float]):
    H, W = img_np.shape[:2]
    ry1, ry2, rx1, rx2 = roi
    y1, y2 = int(H*ry1), int(H*ry2)
    x1, x2 = int(W*rx1), int(W*rx2)
    return img_np[y1:y2, x1:x2], (x1, y1), (W, H)

def _xyxy_from_poly(box):
    # box: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    return [min(xs), min(ys), max(xs), max(ys)]

def _clip_xyxy(x1,y1,x2,y2,W,H):
    x1 = max(0, min(int(x1), W-1))
    y1 = max(0, min(int(y1), H-1))
    x2 = max(0, min(int(x2), W-1))
    y2 = max(0, min(int(y2), H-1))
    return [x1,y1,x2,y2]

def _dedup_text_boxes(texts, boxes, iou_thr=0.5):
    keep_t, keep_b = [], []
    for t, b in zip(texts, boxes):
        dup = False
        for tb, bb in zip(keep_t, keep_b):
            # 동일 텍스트 & IoU 높으면 중복으로 간주
            if t == tb:
                # IoU 계산
                x1 = max(b[0],bb[0]); y1 = max(b[1],bb[1])
                x2 = min(b[2],bb[2]); y2 = min(b[3],bb[3])
                inter = max(0,x2-x1+1) * max(0,y2-y1+1)
                area_b  = (b[2]-b[0]+1)*(b[3]-b[1]+1)
                area_bb = (bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)
                iou = inter / max(1, (area_b + area_bb - inter))
                if iou >= iou_thr:
                    dup = True; break
        if not dup:
            keep_t.append(t); keep_b.append(b)
    return keep_t, keep_b

def check_ocr_box(
    img: np.ndarray,
    display_img: bool = False,
    output_bb_format: str = 'xyxy',
    goal_filtering: str = None,
    easyocr_args: Dict = None,
    use_paddleocr: bool = True,   # PaddleOCR 기본으로 변경
    ocr_engine: str = 'paddleocr',  # PaddleOCR 기본으로 변경
    # ▼ 추가
    roi: Tuple[float,float,float,float] = (0.0, 1.0, 0.0, 1.0),  # 전체 화면으로 테스트
    paddle_det_limit_side_len: int = 1920,
    paddle_rec_score_thresh: float = 0.35,  # 0.7 → 0.35로 낮춤 (짧은 토큰 구제)
    paddle_lang: str = 'korean'
) -> Tuple[List[str], List[List[float]]]:
    """
    OCR을 수행하여 텍스트와 바운딩 박스를 반환합니다.
    
    Args:
        img: 입력 이미지 (numpy array)
        display_img: 이미지 표시 여부
        output_bb_format: 바운딩 박스 형식 ('xyxy' 또는 'xywh')
        goal_filtering: 목표 필터링 텍스트
        easyocr_args: EasyOCR 인수
        use_paddleocr: PaddleOCR 사용 여부 (현재는 EasyOCR만 사용)
        
    Returns:
        Tuple[List[str], List[List[float]]]: (텍스트 리스트, 바운딩 박스 리스트)
    """
    # 디버깅: 입력 이미지 정보 출력
    print(f"[OCR DEBUG] 입력 이미지 타입: {type(img)}")
    if hasattr(img, 'shape'):
        print(f"[OCR DEBUG] 입력 이미지 크기: {img.shape}")
    elif hasattr(img, 'size'):
        print(f"[OCR DEBUG] 입력 이미지 크기: {img.size}")
    
    # 이미지 전처리로 OCR 성능 향상
    if hasattr(img, 'shape'):  # numpy array
        img_np = img.copy()
    else:  # PIL Image
        img_np = np.array(img)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:  # RGB to BGR
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 원본 이미지 크기 저장
    original_height, original_width = img_np.shape[:2]
    print(f"[OCR DEBUG] 전처리 후 이미지 크기: {img_np.shape}")
    
    # 이미지 전처리 단순화 (스케일 변환 제거)
    print(f"[OCR DEBUG] 전처리 시작 - 원본 크기: {img_np.shape}")
    
    # 대비 향상 (CLAHE) - 기본적인 전처리만 유지
    lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    print(f"[OCR DEBUG] 대비 향상 완료")
    
    # 노이즈 제거
    img_np = cv2.medianBlur(img_np, 3)
    print(f"[OCR DEBUG] 노이즈 제거 완료 - 최종 크기: {img_np.shape}")
    
    # 5. 이진화는 제거 (좌표 변환 문제로 인해)
    # 대신 더 강한 대비 향상만 적용
    # gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # img_np = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    """
    OCR을 수행하고 텍스트와 바운딩 박스를 반환합니다.
    
    Args:
        img: 입력 이미지
        display_img: 결과 이미지 표시 여부
        output_bb_format: 바운딩 박스 형식 ('xyxy' 또는 'xywh')
        goal_filtering: 특정 텍스트만 필터링
        easyocr_args: EasyOCR 설정
        use_paddleocr: PaddleOCR 사용 여부
        
    Returns:
        Tuple[List[str], List[List[float]]]: (텍스트 리스트, 바운딩 박스 리스트)
    """
    # EasyOCR 설정
    if easyocr_args is None:
        easyocr_args = {'paragraph': False, 'text_threshold': 0.9}
    
    # OCR 수행
    # OCR 엔진 선택
    print(f"[OCR DEBUG] 선택된 OCR 엔진: {ocr_engine}")
    if ocr_engine == 'tesseract':
        print("[OCR DEBUG] Tesseract OCR 사용")
        return check_ocr_box_tesseract(img, display_img, output_bb_format, goal_filtering)

    elif ocr_engine == 'hybrid':
        # 하이브리드 모드: EasyOCR + Tesseract 조합
        texts1, boxes1 = check_ocr_box_tesseract(img, display_img, output_bb_format, goal_filtering)
        
        # EasyOCR GPU 사용 (CUDA 12.6 호환) - 성능 최적화
        reader = _get_easyocr_reader(langs=('ko','en'), gpu=True, models_dir='./easyocr_models')
        
        # OCR 인수 최적화
        if easyocr_args is None:
            easyocr_args = {
                'detail': 0,  # 상세 정보 비활성화로 속도 향상
                'paragraph': False,  # 단락 모드 비활성화
                'height_ths': 0.5,  # 높이 임계값
                'width_ths': 0.5,   # 너비 임계값
                'text_threshold': 0.6,  # 텍스트 임계값 (높일수록 정확도 향상)
                'link_threshold': 0.4,  # 링크 임계값
                'low_text': 0.3,    # 낮은 텍스트 임계값
                'canvas_size': 2560,  # 캔버스 크기
                'mag_ratio': 1.5,   # 확대 비율 (높일수록 정확도 향상)
            }
        
        result = reader.readtext(img_np, **easyocr_args)
        
        # EasyOCR 결과 파싱
        texts2, boxes2 = [], []
        for (box, text, prob) in result:
            if prob < 0.5:
                continue
            text = text.strip()
            if len(text) < 1:
                continue
            text = re.sub(r'[^\w\s가-힣&()]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) < 1:
                continue
            if goal_filtering is None or goal_filtering in text:
                texts2.append(text)
                # EasyOCR box 형식: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] -> [x1,y1,x2,y2] (단순화)
                x_coords = [point[0] for point in box]
                y_coords = [point[1] for point in box]
                
                # 단순히 min/max로 바운딩 박스 생성
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                
                # 좌표가 원본 이미지 범위를 벗어나지 않도록 제한
                x1 = max(0, min(x1, original_width - 1))
                y1 = max(0, min(y1, original_height - 1))
                x2 = max(0, min(x2, original_width - 1))
                y2 = max(0, min(y2, original_height - 1))
                
                boxes2.append([x1, y1, x2, y2])
        
        # 결과 병합 (중복 제거)
        all_texts = texts1 + texts2
        all_boxes = boxes1 + boxes2
        
        # 중복 제거 (유사한 텍스트는 하나만 유지)
        unique_texts = []
        unique_boxes = []
        for i, text in enumerate(all_texts):
            is_duplicate = False
            for existing_text in unique_texts:
                if text.lower() == existing_text.lower() or text in existing_text or existing_text in text:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_texts.append(text)
                unique_boxes.append(all_boxes[i])
        
        return unique_texts, unique_boxes
    elif ocr_engine == 'paddleocr' or use_paddleocr:
        try:
            print("[PaddleOCR] 싱글톤 인스턴스 사용...")
            ocr = get_paddle_ocr()  # 싱글톤으로 초기화

            # 0) 입력을 numpy BGR로 통일
            if hasattr(img, 'shape'):
                img_np = img.copy()
            else:
                img_np = np.array(img)
                if img_np.ndim == 3 and img_np.shape[2] == 3:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            H0, W0 = img_np.shape[:2]
            print(f"[PaddleOCR] 원본 이미지 크기: {W0}x{H0}")

            # (1) ROI 크롭 (원본 좌표계)
            roi_img, (offx, offy), (W, H) = _crop_roi(img_np, roi)
            print(f"[PaddleOCR] ROI 크롭: {roi_img.shape}, 오프셋 ({offx}, {offy})")

            # (2) 스케일업 – 작은 글자 보강
            scale = 1.5
            proc = cv2.resize(roi_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            print(f"[PaddleOCR] 스케일업: {proc.shape} (scale={scale})")

            # (3) PaddleOCR 실행 (3.x predict / 2.x ocr 둘 다 대응)
            def run_paddle_any(ocr_ins, npimg):
                if hasattr(ocr_ins, "predict"):
                    return ocr_ins.predict(
                        npimg,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                        text_det_limit_type='max',
                        text_det_limit_side_len=paddle_det_limit_side_len,
                        text_rec_score_thresh=paddle_rec_score_thresh
                    )
                else:
                    return ocr_ins.ocr(npimg, cls=True)

            result = run_paddle_any(ocr, proc)
            print(f"[PaddleOCR] OCR 결과: {type(result)}, 길이: {len(result) if result else 0}")

            texts, boxes = [], []
            if result is None:
                return [], []

            # 3.x 형태 감지
            if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict) and 'rec_texts' in result[0]:
                rec_texts = result[0].get('rec_texts', [])
                rec_polys = result[0].get('rec_polys', [])
                rec_scores = result[0].get('rec_scores', [])
                for i, (t, poly) in enumerate(zip(rec_texts, rec_polys)):
                    if i < len(rec_scores) and rec_scores[i] < 0.35:  # 짧은 토큰 구제
                        continue
                    t = re.sub(r'[^\w\s가-힣&()]','', t).strip()
                    if not t: 
                        continue
                    if goal_filtering is None or goal_filtering in t:
                        x1,y1,x2,y2 = _xyxy_from_poly(poly)
                        # (4) 좌표 복원: /scale + ROI 오프셋
                        x1 = x1/scale + offx; x2 = x2/scale + offx
                        y1 = y1/scale + offy; y2 = y2/scale + offy
                        box_xyxy = _clip_xyxy(x1,y1,x2,y2,W0,H0)
                        texts.append(t); boxes.append(box_xyxy)
                        print(f"[PaddleOCR] 텍스트: '{t}' -> 박스: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
            else:
                # 2.x: result = [ [ [poly, (text, score)], ... ] ]
                lines = result[0] if (isinstance(result, list) and len(result)>0) else []
                for line in lines:
                    poly, (t, conf) = line
                    if conf < 0.35:  # 짧은 토큰 구제
                        continue
                    t = re.sub(r'[^\w\s가-힣&()]','', t).strip()
                    if not t: 
                        continue
                    if goal_filtering is None or goal_filtering in t:
                        x1,y1,x2,y2 = _xyxy_from_poly(poly)
                        x1 = x1/scale + offx; x2 = x2/scale + offx
                        y1 = y1/scale + offy; y2 = y2/scale + offy
                        box_xyxy = _clip_xyxy(x1,y1,x2,y2,W0,H0)
                        texts.append(t); boxes.append(box_xyxy)
                        print(f"[PaddleOCR] 텍스트: '{t}' -> 박스: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")

            # 중복 제거
            texts, boxes = _dedup_text_boxes(texts, boxes)
            print(f"[PaddleOCR] 최종 결과: {len(texts)}개 텍스트")

            # 바운딩 박스 형식 변환
            if output_bb_format == 'xywh':
                boxes = [convert_xyxy_to_xywh(b) for b in boxes]
            return texts, boxes

        except Exception as e:
            print(f"[PaddleOCR ERROR] {e}")
            print("PaddleOCR 실패, EasyOCR로 폴백...")
            ocr_engine = 'easyocr'
    
    # EasyOCR 실행(폴백 포함)
    result = []  # 기본값 설정
    if ocr_engine == 'easyocr':
        print("[OCR DEBUG] EasyOCR 사용")
        reader = _get_easyocr_reader(langs=('ko','en'), gpu=True, models_dir='./easyocr_models')
        
        # img_np 재정의 (PaddleOCR try-except 밖에서 사용하기 위해)
        if hasattr(img, 'shape'):
            img_np = img.copy()
        else:
            img_np = np.array(img)
            if img_np.ndim == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Paddle과 동일하게 ROI 적용해서 좌표 일치 보장
        roi_img, (offx, offy), (W, H) = _crop_roi(img_np, roi)
        print(f"[EasyOCR] ROI 크롭: 원본 크기 {img_np.shape} -> ROI 크기 {roi_img.shape}, 오프셋 ({offx}, {offy})")
        
        # (네가 쓰던 easyocr_args 그대로 사용)
        if easyocr_args is None:
            easyocr_args = {'paragraph': False, 'text_threshold': 0.9}
        
        result = reader.readtext(roi_img, **easyocr_args)
    
    # 결과 파싱 및 후처리
    texts = []
    boxes = []
    print(f"[OCR DEBUG] EasyOCR 결과 개수: {len(result)}")
    
    for (box, text, prob) in result:
        if prob < 0.35:  # 0.3 → 0.35로 낮춤 (짧은 토큰 구제)
            continue
        text = re.sub(r'[^\w\s가-힣&()]', '', text).strip()
        if not text:
            continue

        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)

        # ★ ROI 오프셋 보정 ★
        x1 = max(0, min(x1 + offx, W - 1))
        y1 = max(0, min(y1 + offy, H - 1))
        x2 = max(0, min(x2 + offx, W - 1))
        y2 = max(0, min(y2 + offy, H - 1))

        texts.append(text)
        boxes.append([x1, y1, x2, y2])
    
    # 바운딩 박스 형식 변환
    if output_bb_format == 'xywh':
        boxes = [convert_xyxy_to_xywh(box) for box in boxes]
    
    # 디버그 출력 (OCR 결과 확인)
    if len(texts) > 0:
        print(f"[OCR DEBUG] 감지된 텍스트 {len(texts)}개:")
        for i, (text, box) in enumerate(zip(texts[:20], boxes[:20])):  # 처음 20개 출력
            if output_bb_format == 'xyxy':
                x1, y1, x2, y2 = box
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                width, height = x2 - x1, y2 - y1
                print(f"  {i+1:2d}. '{text}' at ({center_x:.1f}, {center_y:.1f}) size({width:.1f}x{height:.1f}) [box: {x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
            else:
                print(f"  {i+1:2d}. '{text}' at box: {box}")
        if len(texts) > 20:
            print(f"  ... 외 {len(texts)-20}개 더")
    else:
        print("[OCR DEBUG] 감지된 텍스트 없음")
    
    return texts, boxes

def check_ocr_box_tesseract(
    img: np.ndarray,
    display_img: bool = False,
    output_bb_format: str = 'xyxy',
    goal_filtering: str = None
) -> Tuple[List[str], List[List[float]]]:
    """
    Tesseract OCR을 사용하여 텍스트와 바운딩 박스를 반환합니다.
    
    Args:
        img: 입력 이미지 (numpy array)
        display_img: 이미지 표시 여부
        output_bb_format: 바운딩 박스 형식 ('xyxy' 또는 'xywh')
        goal_filtering: 목표 필터링 텍스트
        
    Returns:
        Tuple[List[str], List[List[float]]]: (텍스트 리스트, 바운딩 박스 리스트)
    """
    # 이미지 전처리
    if hasattr(img, 'shape'):  # numpy array
        img_np = img.copy()
    else:  # PIL Image
        img_np = np.array(img)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:  # RGB to BGR
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 1. 그레이스케일 변환
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_np
    
    # 2. 노이즈 제거
    gray = cv2.medianBlur(gray, 3)
    
    # 3. 이진화
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Tesseract OCR 설정 - 영어와 한글 모두 인식하도록 최적화
    custom_config = r'--oem 3 --psm 6 -l kor+eng -c tessedit_do_invert=0'
    
    try:
        # Tesseract OCR 수행
        data = pytesseract.image_to_data(binary, config=custom_config, output_type=pytesseract.Output.DICT)
        
        texts = []
        boxes = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            # 신뢰도 필터링 (10 이상만) - 더 많은 텍스트 감지
            if conf < 10 or len(text) < 1:  # 30에서 10으로 더 낮춤
                continue
                
            # 텍스트 정제
            text = re.sub(r'[^\w\s가-힣&()]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if len(text) < 1:
                continue
                
            # 목표 필터링
            if goal_filtering is None or goal_filtering in text:
                texts.append(text)
                
                # 바운딩 박스 추출
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]
                
                if output_bb_format == 'xyxy':
                    box = [x, y, x + w, y + h]
                else:  # xywh
                    box = [x, y, w, h]
                    
                boxes.append(box)
                
    except Exception as e:
        print(f"Tesseract OCR 오류: {e}")
        return [], []
    
    return texts, boxes

def get_yolo_model(model_path: str) -> Any:
    """
    YOLO 모델을 로드합니다.
    
    Args:
        model_path: 모델 파일 경로

    Returns:
        YOLO 모델 객체
    """
    try:
        # 지연 import
        from ultralytics import YOLO
        # YOLOv8 모델 직접 로드
        model = YOLO(os.path.abspath(model_path))
        print(f"YOLO 모델 로드 성공: {os.path.abspath(model_path)}")
        return model
    except Exception as e:
        print(f"YOLO 모델 로드 실패: {e}")
        print(f"모델 경로: {os.path.abspath(model_path)}")
        return None

def get_caption_model_processor(model_name: str = "blip",
                              model_name_or_path: str = "Salesforce/blip-image-captioning-base") -> Tuple[Any, Any]:
    """
    이미지 캡셔닝 모델과 프로세서를 로드합니다.
    
    Args:
        model_name: 모델 이름
        model_name_or_path: 모델 경로 또는 이름
        
    Returns:
        Tuple[Any, Any]: (모델, 프로세서)
    """
    try:
        # 지연 import
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch
        # 온라인 모델 설정 파일 사용
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # 로컬 가중치 파일이 있으면 로드
        local_weights_path = os.path.join("weights", "icon_caption_blip", "pytorch_model.bin")
        if os.path.exists(local_weights_path):
            print(f"로컬 BLIP 가중치 로드: {local_weights_path}")
            state_dict = torch.load(local_weights_path)
            model.load_state_dict(state_dict)
            
        return model, processor
    except Exception as e:
        print(f"캡셔닝 모델 로드 실패: {e}")
        return None, None

def get_som_labeled_img(image: Image.Image,
                        yolo_model: Any,
                        BOX_TRESHOLD: float = 0.05,
                        output_coord_in_ratio: bool = True,
                        ocr_bbox: List = None,
                        draw_bbox_config: Dict = None,
                        caption_model_processor: Tuple = None,
                        ocr_text: List = None,
                        iou_threshold: float = 0.1,
                        imgsz: int = 480) -> Tuple[str, List, List]:
    """
    이미지에 객체 탐지 결과를 레이블링합니다.
    
    Args:
        image: PIL Image 객체
        yolo_model: YOLO 모델
        BOX_TRESHOLD: 바운딩 박스 임계값
        output_coord_in_ratio: 좌표를 비율로 출력할지 여부
        ocr_bbox: OCR 바운딩 박스
        draw_bbox_config: 바운딩 박스 그리기 설정
        caption_model_processor: 캡셔닝 모델과 프로세서
        ocr_text: OCR 텍스트
        iou_threshold: IOU 임계값
        imgsz: 이미지 크기
        
    Returns:
        Tuple[str, List, List]: (레이블링된 이미지 base64, 좌표 리스트, 파싱된 콘텐츠 리스트)
    """
    # 이미지 크기 조정
    image = image.resize((imgsz, imgsz))
    
    # YOLO 추론
    raw_results = yolo_model(image)
    
    # --- API 버전별 결과 형식 통합 처리 ---
    # 1) 결과가 리스트라면 첫 번째 요소로
    res0 = raw_results[0] if isinstance(raw_results, list) else raw_results
    
    # 2) ultralytics YOLOv8: res0.boxes.xyxy
    if hasattr(res0, "boxes"):
        xyxy_tensor = res0.boxes.xyxy
        scores = res0.boxes.conf.cpu().numpy()
        class_ids = res0.boxes.cls.cpu().numpy().astype(int)
    # 3) legacy YOLOv5/YOLOv3: res0.xyxy
    elif hasattr(res0, "xyxy"):
        xyxy_tensor = res0.xyxy[0]
        scores = xyxy_tensor[:, 4].cpu().numpy()
        if xyxy_tensor.shape[1] >= 6:
            class_ids = xyxy_tensor[:, 5].cpu().numpy().astype(int)
        else:
            class_ids = np.full_like(scores, -1, dtype=int)
    else:
        raise RuntimeError(f"Unsupported YOLO result format: {type(res0)}")
    
    # numpy 배열로 변환
    boxes = xyxy_tensor.cpu().numpy()
    
    # 임계값 이상의 박스만 선택
    mask = scores > BOX_TRESHOLD
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    # 좌표를 비율로 변환
    if output_coord_in_ratio:
        boxes[:, [0, 2]] /= imgsz
        boxes[:, [1, 3]] /= imgsz
    
    # 바운딩 박스 그리기
    img_np = np.array(image)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 이미지를 base64로 인코딩
    img_pil = Image.fromarray(img_np)
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # 파싱된 콘텐츠 리스트 생성
    parsed_content = []
    for box, conf, cls in zip(boxes, scores, class_ids):
        content = {
            'bbox': box[:4].tolist(),
            'confidence': float(conf),
            'class': int(cls)
        }
        parsed_content.append(content)
    
    return img_str, boxes.tolist(), parsed_content

def convert_xyxy_to_xywh(box: List[float]) -> List[float]:
    """
    바운딩 박스 형식을 xyxy에서 xywh로 변환합니다.
    
    Args:
        box: [x1, y1, x2, y2] 형식의 바운딩 박스
        
    Returns:
        List[float]: [x, y, w, h] 형식의 바운딩 박스
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h] 