import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from ultralytics import YOLO
import supervision as sv
import os
import sys
import torch
from PIL import Image
import base64
from io import BytesIO
import easyocr
from paddleocr import PaddleOCR
from transformers import BlipProcessor, BlipForConditionalGeneration

class ButtonDetector:
    def __init__(self, model_path: str = "weights/icon_detect/model.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        self.model = YOLO(model_path)
        self.box_annotator = sv.BoxAnnotator()
        
    def detect(self, image: np.ndarray) -> List[Dict]:
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

# 전역 인스턴스 생성
button_detector = ButtonDetector()
stamp_detector = StampDetector()

def load_image(image_path: str) -> np.ndarray:
    """이미지 로드"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    return image

def detect_buttons(image: np.ndarray) -> List[Dict]:
    """버튼 감지 (YOLO 모델 사용)"""
    return button_detector.detect(image)

def get_stamp_position(image: np.ndarray) -> Tuple[int, int]:
    """도장 위치 감지"""
    return stamp_detector.detect(image)

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

def check_ocr_box(
    img: np.ndarray,
    display_img: bool = False,
    output_bb_format: str = 'xyxy',
    goal_filtering: str = None,
    easyocr_args: Dict = None,
    use_paddleocr: bool = True
) -> Tuple[List[str], List[List[float]]]:
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
    # 이미지 전처리
    img_np = np.array(img)
    
    # EasyOCR 설정
    if easyocr_args is None:
        easyocr_args = {'paragraph': False, 'text_threshold': 0.9}
    
    # OCR 수행
    if use_paddleocr:
        # PaddleOCR 사용
        ocr = PaddleOCR(
            use_angle_cls=False,
            lang='korean',
            show_log=False,
            det_model_dir=r"C:\paddleocr_cache\whl\det\ml\Multilingual_PP-OCRv3_det_infer",
            rec_model_dir=r"C:\paddleocr_cache\whl\rec\ml\Multilingual_PP-OCRv3_rec_infer",
            cls_model_dir=r"C:\paddleocr_cache\whl\cls\ml\Multilingual_PP-OCRv3_cls_infer"
        )
        result = ocr.ocr(img_np, cls=False)
        
        # 결과 파싱
        texts = []
        boxes = []
        if result:
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                confidence = line[1][1]
                
                if goal_filtering is None or goal_filtering in text:
                    texts.append(text)
                    boxes.append(box)
                        else:
        # EasyOCR 사용
        reader = easyocr.Reader(['ko', 'en'], gpu=False)
        result = reader.readtext(img_np, **easyocr_args)
        
        # 결과 파싱
        texts = []
        boxes = []
        for (box, text, prob) in result:
            if goal_filtering is None or goal_filtering in text:
                texts.append(text)
                boxes.append(box)
    
    # 바운딩 박스 형식 변환
    if output_bb_format == 'xywh':
        boxes = [convert_xyxy_to_xywh(box) for box in boxes]
    
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
        # 온라인 모델 설정 파일 사용
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        # 로컬 가중치 파일이 있으면 로드
        local_weights_path = os.path.join("weights", "icon_caption_blip", "pytorch_model.bin")
        if os.path.exists(local_weights_path):
            print(f"로컬 BLIP 가중치 로드: {local_weights_path}")
            state_dict = torch.load(local_weights_path, map_location='cpu')
            # 호환성 문제 해결을 위한 옵션 추가
            model.load_state_dict(state_dict, strict=False)
            
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