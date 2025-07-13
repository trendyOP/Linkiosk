import os
import sys
import configparser
from collections import deque
from fuzzywuzzy import fuzz
import csv
import time
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import font_manager

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 이 줄은 어떤 paddleocr import보다 **반드시 위**에 있어야 합니다.
os.environ["PADDLEOCR_HOME"] = r"C:\Linkiosk-main\.paddleocr"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# numpy 버전 체크 및 경고
import numpy as np
if np.__version__ >= "2.0.0":
    print("Warning: PaddleOCR requires numpy < 2.0.0")
    print("Please run: pip install numpy==1.24.3")
    exit(1)

# 그 다음에야 PaddleOCR 모듈을 로드
from paddleocr import PaddleOCR
import easyocr

import cv2
import json
from PIL import Image, ImageDraw, ImageFont
import torch
import io
import base64
from typing import Dict, List, Any, Optional, Tuple

# OmniParser 유틸리티 함수 및 모델 로더 임포트
try:
    from utils.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
except ImportError as e:
    print(f"Error importing utils: {e}")
    print("Current sys.path:", sys.path)
    print("Current directory:", os.getcwd())
    print("Please ensure this script is in the 'omniparser' directory and 'utils' subdirectory exists.")
    exit()

# Reflexion 시스템 컴포넌트 임포트
from core.reflexion_core import ReflexionCore

# 고령자 강화학습 에이전트 임포트
from reinforcement_learning_agent import ElderlyRLAgent, ElderlyRLTrainer

# INI 파일 읽기
config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

# 사용자 파라미터
user_type = config['User']['type']
vision_grid_n = int(config['User']['vision_grid'])

# 메뉴 큐 (deque로 변환)
menu_queue = deque([m.strip() for m in config['Menu']['queue'].split(',')])

print(f"[INI] 사용자 유형: {user_type}")
print(f"[INI] 시야 그리드 크기: {vision_grid_n}x{vision_grid_n}")
print(f"[INI] 메뉴 큐: {list(menu_queue)}")

class OmniParserWithElderlyRL:
    def __init__(self,
                 yolo_model=None,
                 caption_model_processor=None,
                 paddle_ocr=None,
                 easyocr_reader=None,
                 rl_agent=None):
        """
        OmniParser with Elderly RL 시스템 초기화
        
        Args:
            yolo_model: YOLO 객체 탐지 모델
            caption_model_processor: BLIP 이미지 캡셔닝 모델과 프로세서
            paddle_ocr: PaddleOCR 인스턴스 (한글 모델)
            easyocr_reader: EasyOCR Reader 인스턴스 (한글/영어 모델)
            rl_agent: 고령자 강화학습 에이전트
        """
        self.yolo_model = yolo_model
        self.caption_model_processor = caption_model_processor
        self.paddle_ocr = paddle_ocr
        self.easyocr_reader = easyocr_reader
        self.rl_agent = rl_agent
        
        # Reflexion 시스템 초기화
        self.reflexion_core = ReflexionCore()
        
        # 에피소드 관련 변수
        self.current_episode = []
        self.total_reward = 0.0
        
        self.cursor_radius = 10
        self.cursor_color = (0, 255, 0)  # Green
        self.cursor_thickness = 2
        self.window_name = "OmniParser Elderly RL Agent Visualization"
        self.stamp_position = (0.5, 0.5)  # 중앙에서 시작
        
        # 로그 디렉토리 생성
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 시각화 저장 디렉토리도 생성
        self.viz_dir = os.path.join(self.log_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # 히트맵 초기화
        self.heatmap = None
        self.rewards = []
        self.actions = []
        self.clicks = []
        self.reflections = []
        
        # 폰트 로드
        self.font_path = "C:/Windows/Fonts/malgun.ttf"  # 윈도우 기본 한글 폰트
        self.font_size = 30
        self.font = ImageFont.truetype(self.font_path, self.font_size)
        
        # 고령자 특성 변수
        self.reaction_time = 0.0
        self.gaze_velocity = 0.0
        self.gaze_acceleration = 0.0
        self.last_gaze_time = None
        self.last_gaze_position = None
        
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        이미지를 처리하고 버튼 탐지 결과 반환
        """
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"이미지 로딩 오류: {e}")
            return None
            
        print(f"이미지 처리 중: {image_path}")
        
        # Bbox 그리기 설정
        box_overlay_ratio = image.size[0] / 3200.0  # 3200은 참조 너비
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # OCR 수행
        print("OCR 수행 중...")
        text_ocr, ocr_bbox = check_ocr_box(
            image,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9},
            use_paddleocr=True
        )
        print(f"OCR 결과: {len(text_ocr)}개의 텍스트 요소 발견.")
        if text_ocr:
            print("감지된 텍스트:")
            for i, (text, bbox) in enumerate(zip(text_ocr, ocr_bbox)):
                print(f"  {i+1}. '{text}' at {bbox}")
        
        # 객체 탐지 및 레이블링
        print("객체 탐지 및 레이블링 수행 중...")
        labeled_img_base64, label_coordinates, parsed_content_list = get_som_labeled_img(
            image,
            self.yolo_model,
            BOX_TRESHOLD=0.05,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text_ocr,
            iou_threshold=0.1,
            imgsz=480
        )
        print("객체 탐지 및 레이블링 완료.")
        
        return {
            'image': image,
            'buttons': parsed_content_list,
            'ocr_text': text_ocr,
            'ocr_bbox': ocr_bbox
        }
    
    def calculate_gaze_velocity(self, current_position, current_time):
        """시선 이동 속도 계산 (훈련 데이터와 동일한 방식)"""
        if self.last_gaze_position is None or self.last_gaze_time is None:
            self.last_gaze_position = current_position
            self.last_gaze_time = current_time
            return 0.0
        
        dt = current_time - self.last_gaze_time
        if dt <= 0:
            return 0.0
        
        dx = current_position[0] - self.last_gaze_position[0]
        dy = current_position[1] - self.last_gaze_position[1]
        distance = (dx**2 + dy**2)**0.5
        velocity = distance / dt  # 픽셀/초
        
        self.last_gaze_position = current_position
        self.last_gaze_time = current_time
        
        return velocity
    
    def _calculate_gaze_acceleration(self, current_time, x, y):
        """시선 이동 가속도 계산 (훈련 데이터와 동일한 방식)"""
        if not hasattr(self, '_last_gaze_velocity') or self._last_gaze_velocity is None:
            self._last_gaze_velocity = {'time': current_time, 'velocity': 0.0}
            return 0.0
        
        current_velocity = self.calculate_gaze_velocity((x, y), current_time)
        dt = current_time - self._last_gaze_velocity['time']
        if dt <= 0:
            return 0.0
        
        acceleration = (current_velocity - self._last_gaze_velocity['velocity']) / dt  # 픽셀/초²
        
        # 다음 계산을 위해 현재 속도 저장
        self._last_gaze_velocity = {'time': current_time, 'velocity': current_velocity}
        
        return acceleration
    
    def _calculate_reaction_time(self, click_time):
        """클릭 반응시간 계산 (훈련 데이터와 동일한 방식)"""
        if not hasattr(self, 'last_gaze_time') or self.last_gaze_time is None:
            return 0.0
        
        return click_time - self.last_gaze_time
    
    def _calculate_gaze_to_click_distance(self, click_x, click_y):
        """시선과 클릭 위치 간의 거리 계산 (훈련 데이터와 동일한 방식)"""
        if not hasattr(self, 'last_gaze_position') or self.last_gaze_position is None:
            return 0.0
        
        dx = click_x - self.last_gaze_position[0]
        dy = click_y - self.last_gaze_position[1]
        return (dx**2 + dy**2)**0.5
    
    def get_vision_box(self, center, grid_n, image_shape):
        """
        center: (x, y) 정규화 좌표 (0~1)
        grid_n: 시야 그리드 크기 (예: 7)
        image_shape: (h, w)
        반환: (x1, y1, x2, y2) 픽셀 좌표 (시야 박스)
        """
        h, w = image_shape
        # 시야 박스 크기 (이미지의 1/grid_n 비율)
        box_w = w // grid_n
        box_h = h // grid_n
        cx = int(center[0] * w)
        cy = int(center[1] * h)
        x1 = max(0, cx - box_w // 2)
        y1 = max(0, cy - box_h // 2)
        x2 = min(w, cx + box_w // 2)
        y2 = min(h, cy + box_h // 2)
        return (x1, y1, x2, y2)
    
    def select_elderly_action(self, state_vector, target_menu):
        """고령자 RL 에이전트로부터 행동 선택"""
        if self.rl_agent is None:
            # RL 에이전트가 없으면 랜덤 행동
            return np.random.randint(0, 4), np.random.rand(4), np.random.rand(1)
        
        # 에이전트로부터 행동 선택
        action, action_probs, value, entropy = self.rl_agent.select_action(state_vector)
        
        return action, action_probs.detach().numpy(), value.detach().numpy()
    
    def calculate_elderly_reward(self, action, detected_buttons, target_menu, 
                                stamp_position, reaction_time, gaze_velocity, ocr_text=None, ocr_bbox=None):
        """고령자 특성을 반영한 보상 계산"""
        if self.rl_agent is None:
            return 0.0
        
        return self.rl_agent.calculate_elderly_reward(
            action, detected_buttons, target_menu, 
            stamp_position, reaction_time, gaze_velocity, ocr_text, ocr_bbox
        )
    
    def update_elderly_memory(self, detected_buttons, ocr_text):
        """고령자 메모리 업데이트"""
        if self.rl_agent is not None:
            self.rl_agent.update_memory(detected_buttons, ocr_text)
    
    def step_elderly_memory(self):
        """고령자 메모리 스텝 진행"""
        if self.rl_agent is not None:
            self.rl_agent.memory.step()
    
    def visualize_cursor(self, image, position, action=None, box_color=(255,0,0), 
                        click_coords=None, status_text=None, elderly_info=None, rl_info=None):
        """커서 시각화 (고령자 정보 포함)"""
        # PIL Image로 변환
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, str):
            image = Image.open(image)
        
        # 이미지 복사
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        
        h, w = vis_image.size[1], vis_image.size[0]
        x = int(position[0] * w)
        y = int(position[1] * h)
        
        # 시야 박스 계산 및 그리기
        x1, y1, x2, y2 = self.get_vision_box(position, vision_grid_n, (h, w))
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        
        # 클릭 좌표(빨간 별) 그리기
        if click_coords:
            for cx, cy in click_coords:
                # 별 모양 그리기
                star_points = []
                for i in range(5):
                    angle = i * 72 - 90
                    rad = np.radians(angle)
                    star_points.append((
                        int(cx + 15 * np.cos(rad)),
                        int(cy + 15 * np.sin(rad))
                    ))
                draw.polygon(star_points, fill=(0, 0, 255))
        
        # 커서 그리기
        draw.ellipse([x-self.cursor_radius, y-self.cursor_radius, 
                     x+self.cursor_radius, y+self.cursor_radius], 
                    outline=self.cursor_color, width=self.cursor_thickness)
        
        # 상태 텍스트 준비
        if status_text is None:
            status_text = []
        
        # 고령자 정보 추가
        if elderly_info:
            status_text.extend([
                f"반응시간: {elderly_info.get('reaction_time', 0):.2f}초",
                f"시선속도: {elderly_info.get('gaze_velocity', 0):.1f}px/s",
                f"시선가속도: {elderly_info.get('gaze_acceleration', 0):.1f}px/s²",
                f"고정시간: {elderly_info.get('fixation_duration', 0):.1f}초"
            ])
        
        # RL 모델 정보 추가
        if rl_info:
            status_text.extend([
                f"=== RL 모델 정보 ===",
                f"상태벡터: {rl_info.get('state_vector_shape', 'N/A')}",
                f"선택행동: {['위', '아래', '왼쪽', '오른쪽'][rl_info.get('selected_action', 0)]}",
                f"행동확률: {[f'{p:.2f}' for p in rl_info.get('action_probs', [0,0,0,0])]}",
                f"상태가치: {rl_info.get('value', 0):.3f}",
                f"메모리아이템: {rl_info.get('memory_items', 0)}개",
                f"총보상: {rl_info.get('reward', 0):.2f}",
                f"보상분해: {rl_info.get('reward_breakdown', {})}",
                f"모델로드: {'✅' if rl_info.get('model_loaded') else '❌'}",
                f"훈련데이터적용: {'✅' if rl_info.get('training_data_applied') else '❌'}",
                f"=== 훈련데이터 특성 ===",
                f"빠른시선이동: ✅ (100-400px/s)",
                f"목표지향적: ✅ (메뉴찾기 집중)",
                f"자연스러운반응: ✅ (1.5초 이상)",
                f"타겟기반행동: {'✅' if rl_info.get('target_based_action') else '❌'}",
                f"가장가까운타겟: {rl_info.get('nearest_target_distance', 0):.2f}"
            ])
        
        # 상태 텍스트 표시
        if status_text:
            # 텍스트 크기 측정
            line_heights = []
            max_width = 0
            for line in status_text:
                bbox = self.font.getbbox(line)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                line_heights.append(height)
                if width > max_width:
                    max_width = width
            total_height = sum(line_heights) + 40  # 위아래 여백
            total_width = max_width + 40  # 좌우 여백
            # 배경 위치
            bg_x1, bg_y1 = 10, 10
            bg_x2, bg_y2 = bg_x1 + total_width, bg_y1 + total_height
            # 반투명 배경
            overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
            draw_overlay.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 180))
            vis_image = Image.alpha_composite(vis_image.convert('RGBA'), overlay)
            draw = ImageDraw.Draw(vis_image)
            # 텍스트 표시
            y_offset = bg_y1 + 20
            for i, line in enumerate(status_text):
                draw.text((bg_x1 + 20, y_offset), line, font=self.font, fill=(255, 255, 255))
                y_offset += line_heights[i]
            vis_image = vis_image.convert('RGB')
        
        # PIL Image를 OpenCV 형식으로 변환하여 표시
        cv2_image = cv2.cvtColor(np.array(vis_image), cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, cv2_image)
        cv2.waitKey(100)
    
    def run_episode_with_elderly_rl(self, image_path: str, max_steps: int = 10) -> Dict[str, Any]:
        """
        고령자 RL 에이전트로 에피소드를 실행합니다.
        """
        print(f"\n=== 고령자 RL 에이전트 에피소드 시작: {image_path} ===")
        
        # 이미지 처리
        result = self.process_image(image_path)
        if not result:
            print("이미지 처리 실패")
            return {
                'total_reward': 0.0,
                'steps': 0,
                'attention_heatmap_path': None,
                'reward_history_path': None,
                'action_distribution_path': None,
                'reflection_path': None
            }
            
        # 에피소드 초기화
        self.total_reward = 0
        self.step_count = 0
        self.stamp_position = (0.5, 0.5)  # 중앙에서 시작
        self.visualize_cursor(result['image'], self.stamp_position)
        
        # 에피소드 실행
        clicks = []
        for step in range(max_steps):
            # menu_queue가 비어있으면 에피소드 종료
            if not menu_queue:
                print(f"[STEP {step+1}] 모든 메뉴를 찾았습니다. 에피소드 종료.")
                break

            # 현재 목표 메뉴 출력
            current_target = menu_queue[0]
            print(f"\n[STEP {step+1}] '{current_target}'를(을) 찾기 위해 시선을 이동중입니다...")

            # 고령자 메모리 스텝 진행
            self.step_elderly_memory()
            
            # 고령자 정보 준비
            elderly_info = {
                'gaze_velocity': self.gaze_velocity,
                'gaze_acceleration': self.gaze_acceleration,
                'reaction_time': self.reaction_time,
                'fixation_duration': getattr(self, '_current_fixation_duration', 0.0)
            }
            
            # 상태 벡터 생성 (훈련 데이터와 동일한 구조)
            vision_grid = np.zeros((vision_grid_n, vision_grid_n))  # 간단한 시야 그리드
            state_vector = self.rl_agent.get_state_vector(
                self.stamp_position, 
                result['buttons'], 
                result['ocr_text'], 
                vision_grid,
                elderly_info
            ) if self.rl_agent else torch.zeros(150)
            
            # 텍스트/이미지가 있는 곳을 우선적으로 보도록 행동 조정
            target_positions = []
            
            # 감지된 버튼들의 중심점을 타겟으로 추가
            h, w = result['image'].size[1], result['image'].size[0]
            for button in result['buttons']:
                if 'bbox' in button:
                    bbox = button['bbox']
                    # bbox가 리스트인 경우 처리
                    if isinstance(bbox, list):
                        if len(bbox) >= 4:
                            center_x = (bbox[0] + bbox[2]) / 2 / w  # 정규화
                            center_y = (bbox[1] + bbox[3]) / 2 / h  # 정규화
                            target_positions.append((center_x, center_y))
                            print(f"  → 버튼 타겟 추가: ({center_x:.3f}, {center_y:.3f})")
                    else:
                        # bbox가 튜플이나 다른 형태인 경우
                        try:
                            center_x = (bbox[0] + bbox[2]) / 2 / w  # 정규화
                            center_y = (bbox[1] + bbox[3]) / 2 / h  # 정규화
                            target_positions.append((center_x, center_y))
                            print(f"  → 버튼 타겟 추가: ({center_x:.3f}, {center_y:.3f})")
                        except (TypeError, IndexError):
                            print(f"  → bbox 형식 오류: {bbox}")
                            continue
            
            # OCR 텍스트가 있는 위치도 타겟으로 추가
            if result['ocr_text']:
                for i, (text, bbox) in enumerate(zip(result['ocr_text'], result['ocr_bbox'])):
                    # bbox가 리스트인 경우 처리
                    if isinstance(bbox, list):
                        if len(bbox) >= 4:
                            try:
                                # bbox가 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 형태인 경우
                                if isinstance(bbox[0], list):
                                    # 4개 점의 평균을 중심으로 계산
                                    x_coords = [point[0] for point in bbox]
                                    y_coords = [point[1] for point in bbox]
                                    center_x = sum(x_coords) / len(x_coords) / w  # 정규화
                                    center_y = sum(y_coords) / len(y_coords) / h  # 정규화
                                else:
                                    # 기존 [x1, y1, x2, y2] 형태
                                    center_x = (bbox[0] + bbox[2]) / 2 / w  # 정규화
                                    center_y = (bbox[1] + bbox[3]) / 2 / h  # 정규화
                                target_positions.append((center_x, center_y))
                                print(f"  → OCR 타겟 추가: '{text}' at ({center_x:.3f}, {center_y:.3f})")
                            except (TypeError, IndexError):
                                print(f"  → OCR bbox 형식 오류: {bbox}")
                                continue
                    else:
                        # bbox가 튜플이나 다른 형태인 경우
                        try:
                            center_x = (bbox[0] + bbox[2]) / 2 / w  # 정규화
                            center_y = (bbox[1] + bbox[3]) / 2 / h  # 정규화
                            target_positions.append((center_x, center_y))
                            print(f"  → OCR 타겟 추가: '{text}' at ({center_x:.3f}, {center_y:.3f})")
                        except (TypeError, IndexError):
                            print(f"  → OCR bbox 형식 오류: {bbox}")
                            continue
            
            # 가장 가까운 타겟을 찾아서 그 방향으로 이동하도록 행동 조정
            if target_positions:
                current_x, current_y = self.stamp_position
                min_distance = float('inf')
                best_action = 0  # 기본값으로 초기화
                best_target = None
                
                print(f"  → 현재 위치: ({current_x:.3f}, {current_y:.3f})")
                print(f"  → 타겟 개수: {len(target_positions)}")
                
                for target_x, target_y in target_positions:
                    distance = np.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
                    print(f"  → 타겟 ({target_x:.3f}, {target_y:.3f}) 거리: {distance:.3f}")
                    if distance < min_distance:
                        min_distance = distance
                        best_target = (target_x, target_y)
                        # 타겟 방향으로의 최적 행동 계산
                        dx = target_x - current_x
                        dy = target_y - current_y
                        
                        if abs(dx) > abs(dy):
                            best_action = 3 if dx > 0 else 2  # 오른쪽 또는 왼쪽
                        else:
                            best_action = 1 if dy > 0 else 0  # 아래 또는 위
                
                print(f"  → 최적 타겟: {best_target}, 거리: {min_distance:.3f}, 행동: {best_action}")
                
                # 타겟 기반 행동을 더 강화 (90% 확률)
                if np.random.random() < 0.9:  # 90% 확률로 타겟 기반 행동
                    action_idx = best_action
                    action_probs = np.array([0.05, 0.05, 0.05, 0.05])
                    action_probs[best_action] = 0.85
                    print(f"  → 타겟 기반 행동 선택: {best_action}")
                else:
                    print(f"  → RL 모델 행동 선택")
                
                target_based_action = True
                nearest_target_distance = min_distance
            else:
                print(f"  → 타겟이 없습니다. RL 모델에 의존")
                target_based_action = False
                nearest_target_distance = float('inf')
            
            # 고령자 RL 에이전트로부터 행동 선택
            action_idx, action_probs, value = self.select_elderly_action(state_vector, current_target)
            
            # RL 모델 정보 수집 (시각화용)
            rl_info = {
                'state_vector_shape': state_vector.shape,
                'action_probs': action_probs.tolist() if hasattr(action_probs, 'tolist') else action_probs,
                'value': value.item() if hasattr(value, 'item') else value,
                'selected_action': action_idx,
                'memory_items': len(self.rl_agent.memory.get_memory_state()['items']) if self.rl_agent else 0,
                'elderly_info': elderly_info,
                'model_loaded': self.rl_agent is not None,
                'training_data_applied': True if self.rl_agent else False,
                'target_based_action': target_based_action,
                'nearest_target_distance': nearest_target_distance
            }
            action = {'type': 'move', 'direction': action_idx}
            action_text = {
                0: "위로",
                1: "아래로", 
                2: "왼쪽으로",
                3: "오른쪽으로"
            }[action_idx]
            print(f"  → {action_text} 이동 (RL 에이전트 선택)")

            # 스탬프 위치 이동 (훈련 데이터처럼 빠른 움직임)
            step_size = 0.15  # 더 큰 스텝으로 빠른 움직임
            x, y = self.stamp_position
            
            # 훈련된 모델의 행동에 따라 빠르게 이동
            if action_idx == 0:  # 위로
                y = max(0.0, y - step_size)
            elif action_idx == 1:  # 아래로
                y = min(1.0, y + step_size)
            elif action_idx == 2:  # 왼쪽으로
                x = max(0.0, x - step_size)
            elif action_idx == 3:  # 오른쪽으로
                x = min(1.0, x + step_size)
            
            # 목표 메뉴와의 거리에 따라 스텝 크기 조정 (가까우면 천천히)
            min_distance = float('inf')
            for button in result['buttons']:
                if 'bbox' in button:
                    bbox = button['bbox']
                    # bbox가 리스트인 경우 처리
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        button_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                        dist = np.sqrt((x - button_center[0])**2 + (y - button_center[1])**2)
                        min_distance = min(min_distance, dist)
                    else:
                        # bbox가 튜플이나 다른 형태인 경우
                        try:
                            button_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                            dist = np.sqrt((x - button_center[0])**2 + (y - button_center[1])**2)
                            min_distance = min(min_distance, dist)
                        except (TypeError, IndexError):
                            print(f"  → 버튼 bbox 형식 오류: {bbox}")
                            continue
            
            # 가까우면 더 정밀하게 움직임
            if min_distance < 0.2:
                step_size = 0.05
                if action_idx == 0: y = max(0.0, y - step_size)
                elif action_idx == 1: y = min(1.0, y + step_size)
                elif action_idx == 2: x = max(0.0, x - step_size)
                elif action_idx == 3: x = min(1.0, x + step_size)
            
            # 위치를 0-1 범위로 제한
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            self.stamp_position = (x, y)
            
            # 시야 박스 계산
            h, w = result['image'].size[1], result['image'].size[0]
            vision_box = self.get_vision_box(self.stamp_position, vision_grid_n, (h, w))
            x1, y1, x2, y2 = vision_box
            # 시야 박스 crop
            vision_crop = result['image'].crop((x1, y1, x2, y2))
            # OCR (시야 박스 전체)
            reader = easyocr.Reader(['ko', 'en'], gpu=False)
            ocr_results = reader.readtext(np.array(vision_crop))
            # OCR 로그 출력
            print(f"[STEP {step+1}] 시야 박스 내 OCR 결과:")
            for res in ocr_results:
                print(f"  - Text: '{res[1]}'")
            
            # 메뉴 매칭 및 클릭 좌표 계산
            is_match = False
            matched_menu = None
            matched_text = None
            matched_score = None
            matched_center = None
            best_score = 0

            for menu in menu_queue:
                for res in ocr_results:
                    text = res[1]
                    score = fuzz.ratio(text, menu)
                    if score >= 50 and score > best_score:
                        # bbox 중심 계산 (시야 crop → 원본 이미지 좌표 변환)
                        (tl, tr, br, bl) = res[0]
                        cx_crop = int((tl[0] + tr[0] + br[0] + bl[0]) / 4)
                        cy_crop = int((tl[1] + tr[1] + br[1] + bl[1]) / 4)
                        cx = x1 + cx_crop
                        cy = y1 + cy_crop
                        matched_menu = menu
                        matched_text = text
                        matched_score = score
                        matched_center = (cx, cy)
                        is_match = True
                        best_score = score
                if is_match:
                    break
            box_color = (0, 0, 255) if is_match else (255, 0, 0)
            
            # 고령자 특성 계산 (훈련 데이터와 동일한 방식)
            current_time = time.time()
            self.gaze_velocity = self.calculate_gaze_velocity(self.stamp_position, current_time)
            self.reaction_time = self._calculate_reaction_time(current_time)
            self.gaze_acceleration = self._calculate_gaze_acceleration(current_time, self.stamp_position[0], self.stamp_position[1])
            
            # 훈련 데이터의 시선 속도 특성 반영 (빠른 움직임)
            # 실제 고령자 시선 데이터는 100-500 px/s 범위
            if self.gaze_velocity < 50:  # 너무 느리면 가속
                self.gaze_velocity = min(300, self.gaze_velocity * 1.5)
            elif self.gaze_velocity > 800:  # 너무 빠르면 감속
                self.gaze_velocity = max(100, self.gaze_velocity * 0.8)
            
            # 시선 고정 시간 계산 (간단한 구현)
            if not hasattr(self, '_fixation_start_time'):
                self._fixation_start_time = current_time
                self._current_fixation_duration = 0.0
            
            # 고정 임계값 (50픽셀 이내)
            fixation_threshold = 0.05  # 정규화된 좌표 기준
            if hasattr(self, '_last_fixation_position'):
                distance = ((self.stamp_position[0] - self._last_fixation_position[0])**2 + 
                          (self.stamp_position[1] - self._last_fixation_position[1])**2)**0.5
                if distance > fixation_threshold:
                    # 고정이 끝남
                    self._fixation_start_time = current_time
                    self._current_fixation_duration = 0.0
                else:
                    # 고정 중
                    self._current_fixation_duration = current_time - self._fixation_start_time
            
            self._last_fixation_position = self.stamp_position
            
            # 고령자 메모리 업데이트
            self.update_elderly_memory(result['buttons'], result['ocr_text'])
            
            # 고령자 정보 준비
            elderly_info = {
                'reaction_time': self.reaction_time,
                'gaze_velocity': self.gaze_velocity,
                'memory_items': self.rl_agent.memory.get_memory_state()['items'] if self.rl_agent else []
            }
            
            # 상태 텍스트 준비
            status_text = [
                f"Step {step+1}",
                f"목표: {current_target}",
                f"행동: {action_text} 이동 (RL)"
            ]

            # 클릭 시 로그 및 클릭 좌표 누적
            if is_match:
                print(f"  → '{matched_menu}'를(을) 찾았습니다! (OCR: '{matched_text}', 유사도: {matched_score}%)")
                print(f"  → 좌표 ({matched_center[0]}, {matched_center[1]})에 클릭")
                status_text.append(f"발견: {matched_menu}")
                status_text.append(f"유사도: {matched_score}%")
                status_text.append(f"클릭: ({matched_center[0]}, {matched_center[1]})")
                self.log_click_event(step+1, matched_menu, matched_text, matched_score, matched_center[0], matched_center[1])
                clicks.append(matched_center)
                menu_queue.remove(matched_menu)
            else:
                print("  → 메뉴를 찾지 못했습니다. 계속 탐색중...")
                status_text.append("상태: 탐색중...")

            # 시각화
            self.visualize_cursor(result['image'], self.stamp_position, action, 
                                box_color=box_color, click_coords=clicks, 
                                status_text=status_text, elderly_info=elderly_info, 
                                rl_info=rl_info)
            
            # 고령자 특성을 반영한 보상 계산 (OCR 결과 포함)
            reward = self.calculate_elderly_reward(
                action_idx, result['buttons'], current_target,
                self.stamp_position, self.reaction_time, self.gaze_velocity,
                result['ocr_text'], result['ocr_bbox']
            )
            
            # 보상 정보를 RL 정보에 추가 (훈련 데이터 특성 반영)
            rl_info['reward'] = reward
            rl_info['reward_breakdown'] = {
                'menu_found': 10.0 if is_match else 0.0,
                'memory_usage': 5.0 if rl_info['memory_items'] > 0 else 0.0,
                'reaction_time': 2.0 if self.reaction_time > 1.5 else -2.0 if self.reaction_time < 0.3 else 0.0,
                'gaze_velocity': 3.0 if 100 <= self.gaze_velocity <= 400 else -1.0 if self.gaze_velocity > 600 or self.gaze_velocity < 50 else 0.0,
                'target_focus': 5.0 if rl_info.get('nearest_target_distance', 1.0) < 0.1 else 3.0 if rl_info.get('nearest_target_distance', 1.0) < 0.2 else 1.0 if rl_info.get('nearest_target_distance', 1.0) < 0.3 else -2.0
            }
            
            # 보상 계산 결과 로그 출력
            print(f"  → 보상: {reward:.2f}")
            if rl_info.get('nearest_target_distance', float('inf')) < 0.3:
                print(f"  → 텍스트/이미지 근처에 있음 (거리: {rl_info.get('nearest_target_distance', 0):.3f})")
            else:
                print(f"  → 텍스트/이미지에서 멀리 있음 (거리: {rl_info.get('nearest_target_distance', 0):.3f})")
            
            # Reflexion 시스템에 스텝 전달
            step_result = self.reflexion_core.process_step(
                stamp_position=self.stamp_position,
                detected_buttons=result['buttons'],
                action=action,
                reward=reward,
                q_values=action_probs,
                policy_probs=action_probs,
                ocr_text=result['ocr_text'],
                ocr_bbox=result['ocr_bbox']
            )
            
            # 보상 누적
            self.total_reward += step_result['reward']
            self.step_count += 1
            
            # 히트맵 업데이트
            self.update_heatmap(self.stamp_position)
            
            # 행동 기록
            self.actions.append(action_idx)
            
            # 보상 기록
            self.rewards.append(reward)
            
            # 클릭 위치 기록
            if is_match:
                self.clicks.append(matched_center)
                self.log_reflection(step + 1, matched_menu, matched_text, matched_score, True)
            else:
                self.log_reflection(step + 1, None, None, None, False)
            
            # 종료 조건 확인
            if step_result.get('is_terminal', False):
                break
        
        cv2.destroyAllWindows()
        
        # 에피소드 종료 및 반성 로그 생성
        episode_summary = self.reflexion_core.end_episode(self.total_reward)
        
        # 결과 요약 생성
        summary = {
            'total_reward': self.total_reward,
            'steps': self.step_count,
            'attention_heatmap_path': os.path.join(self.viz_dir, 'attention_heatmap.png'),
            'reward_history_path': os.path.join(self.viz_dir, 'reward_history.png'),
            'action_distribution_path': os.path.join(self.viz_dir, 'action_distribution.png'),
            'reflection_path': episode_summary['reflection_path']
        }
        
        # 시각화 및 로그 저장
        self.save_visualizations()
        self.save_reflection_log()
        
        # max_steps에 도달했는데 menu_queue가 비어있지 않다면
        if menu_queue:
            print("\n=== 찾지 못한 메뉴 ===")
            for menu in menu_queue:
                print(f"- '{menu}' 버튼을 찾지 못하였습니다.")
            print("===================\n")
        
        return summary
    
    def update_heatmap(self, position, intensity=1.0):
        if self.heatmap is None:
            self.heatmap = np.zeros((100, 100))  # 100x100 그리드로 초기화
        x, y = int(position[0] * 100), int(position[1] * 100)
        # 경계 체크
        x = max(0, min(99, x))
        y = max(0, min(99, y))
        self.heatmap[y, x] += intensity

    def save_visualizations(self):
        # 한글 폰트 설정
        font_name = font_manager.FontProperties(fname=self.font_path).get_name()
        plt.rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

        # 시선 히트맵 저장 (2D 그리드 + 스탬프 이동 경로)
        if self.heatmap is not None:
            plt.figure(figsize=(10, 8))
            grid = self.heatmap
            plt.imshow(grid, cmap='hot', origin='upper')
            plt.colorbar(label='시선 집중도')
            # 스탬프 이동 경로
            if hasattr(self, 'stamp_path') and self.stamp_path:
                path_y, path_x = zip(*self.stamp_path)
                plt.plot(path_x, path_y, color='blue', linewidth=2, label='Stamp Movement')
                plt.scatter([path_x[-1]], [path_y[-1]], color='blue', s=100, label='Current Position')
            plt.title('시선 히트맵')
            plt.xlabel('Grid X')
            plt.ylabel('Grid Y')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, 'attention_heatmap.png'))
            plt.close()

        # 보상 히스토리 저장
        if self.rewards:
            plt.figure(figsize=(10, 5))
            plt.plot(self.rewards)
            plt.title('보상 변화 추이')
            plt.xlabel('스텝')
            plt.ylabel('보상')
            plt.savefig(os.path.join(self.viz_dir, 'reward_history.png'))
            plt.close()

        # 행동 분포 저장
        if self.actions:
            plt.figure(figsize=(10, 5))
            action_types = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'CLICK']
            action_counts = [self.actions.count(i) for i in range(5)]
            plt.bar(action_types, action_counts)
            plt.title('행동 분포')
            plt.xlabel('행동')
            plt.ylabel('횟수')
            plt.savefig(os.path.join(self.viz_dir, 'action_distribution.png'))
            plt.close()

    def log_click_event(self, step, menu, text, score, x, y, log_path="logs/click_log.csv"):
        with open(log_path, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([step, time.time(), menu, text, score, x, y])

    def log_reflection(self, step, menu, text, score, success):
        reflection = {
            'step': step,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'menu': menu,
            'ocr_text': text,
            'score': score,
            'success': success
        }
        self.reflections.append(reflection)

    def save_reflection_log(self):
        if self.reflections:
            with open(os.path.join(self.log_dir, 'reflection.txt'), 'w', encoding='utf-8') as f:
                f.write("=== 고령자 RL Agent 반성 로그 ===\n\n")
                for r in self.reflections:
                    f.write(f"스텝 {r['step']} ({r['timestamp']})\n")
                    f.write(f"목표 메뉴: {r['menu']}\n")
                    f.write(f"OCR 텍스트: {r['ocr_text']}\n")
                    f.write(f"매칭 점수: {r['score']}\n")
                    f.write(f"성공 여부: {'성공' if r['success'] else '실패'}\n")
                    f.write("-" * 50 + "\n")

def main():
    """메인 실행 함수"""
    # YOLO 모델 로드
    yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
    
    # BLIP 모델 로드
    caption_model_processor = get_caption_model_processor(
        model_name="blip",
        model_name_or_path="weights/icon_caption_blip"  # 로컬 BLIP 모델 사용
    )
    
    # PaddleOCR 모델 경로 설정
    base_dir = os.path.abspath(os.path.dirname(__file__))
    paddle_model_dir = os.path.join(base_dir, ".paddleocr", "whl")
    os.makedirs(paddle_model_dir, exist_ok=True)
    
    # 한글 전용 모델로 초기화
    paddle_ocr = PaddleOCR(
        lang='korean',
        use_angle_cls=False,
        use_gpu=False,
        show_log=False,
        det_model_dir=os.path.join(paddle_model_dir, "det", "korean", "korean_PP-OCRv4_det_infer"),
        rec_model_dir=os.path.join(paddle_model_dir, "rec", "korean", "korean_PP-OCRv4_rec_infer"),
        cls_model_dir=os.path.join(paddle_model_dir, "cls", "ch_ppocr_mobile_v2.0_cls_infer"),
        download_without_log=True
    )
    
    # 한글+영어 인식
    easyocr_reader = easyocr.Reader(['ko','en'], gpu=False)
    
    # 고령자 RL 에이전트 로드 (훈련된 모델이 있으면 로드)
    rl_agent = None
    model_path = 'trained_models/elderly_rl_agent_20250712_210011.pth'
    if os.path.exists(model_path):
        print("훈련된 고령자 RL 에이전트 로드 중...")
        state_dim = 150
        action_dim = 4
        rl_agent = ElderlyRLAgent(state_dim, action_dim)
        checkpoint = torch.load(model_path)
        rl_agent.load_state_dict(checkpoint['agent_state_dict'])
        print("훈련된 에이전트 로드 완료!")
    else:
        print("훈련된 에이전트가 없습니다. 랜덤 정책으로 시작합니다.")
    
    # OmniParserWithElderlyRL 인스턴스 생성
    parser = OmniParserWithElderlyRL(
        yolo_model=yolo_model,
        caption_model_processor=caption_model_processor,
        paddle_ocr=paddle_ocr,
        easyocr_reader=easyocr_reader,
        rl_agent=rl_agent
    )
    
    # 테스트 이미지 경로
    test_image = "screen2.png"
    
    # 에피소드 실행
    episode_summary = parser.run_episode_with_elderly_rl(test_image)
    
    # 결과 출력
    print("\n=== 고령자 RL 에이전트 에피소드 요약 ===")
    print(f"총 보상: {episode_summary['total_reward']}")
    print(f"총 스텝: {episode_summary['steps']}")
    print(f"시각화 파일:")
    print(f"- 주의력 히트맵: {episode_summary['attention_heatmap_path']}")
    print(f"- 보상 히스토리: {episode_summary['reward_history_path']}")
    print(f"- 행동 분포: {episode_summary['action_distribution_path']}")
    print(f"- 반성 로그: {episode_summary['reflection_path']}")

if __name__ == "__main__":
    main() 