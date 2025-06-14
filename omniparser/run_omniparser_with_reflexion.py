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

class OmniParserWithReflexion:
    def __init__(self,
                 yolo_model=None,
                 caption_model_processor=None,
                 paddle_ocr=None,
                 easyocr_reader=None):
        """
        OmniParser with Reflexion 시스템 초기화
        
        Args:
            yolo_model: YOLO 객체 탐지 모델
            caption_model_processor: BLIP 이미지 캡셔닝 모델과 프로세서
            paddle_ocr: PaddleOCR 인스턴스 (한글 모델)
            easyocr_reader: EasyOCR Reader 인스턴스 (한글/영어 모델)
        """
        self.yolo_model = yolo_model
        self.caption_model_processor = caption_model_processor
        self.paddle_ocr = paddle_ocr
        self.easyocr_reader = easyocr_reader
        
        # Reflexion 시스템 초기화
        self.reflexion_core = ReflexionCore()
        
        # 에피소드 관련 변수
        self.current_episode = []
        self.total_reward = 0.0
        
        self.cursor_radius = 10
        self.cursor_color = (0, 255, 0)  # Green
        self.cursor_thickness = 2
        self.window_name = "OmniParser Agent Visualization"
        self.stamp_position = (0.5, 0.5)  # 중앙에서 시작
        
        # 로그 디렉토리 생성
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
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
        
    def simulate_agent_step(self, image_path: str) -> Dict[str, Any]:
        """
        에이전트의 단일 스텝을 시뮬레이션합니다.
        """
        # 이미지 처리
        result = self.process_image(image_path)
        if result is None:
            return None
            
        # 현재 스탬프 위치 (예시: 이미지 중앙)
        stamp_position = (0.5, 0.5)
        
        # 시각 상태 초기화 (예시: 10x10 그리드)
        vision_grid = np.zeros((10, 10))
        
        # 버튼 감지 결과
        detected_buttons = result['buttons']
        
        # OCR 결과
        ocr_text = result['ocr_text']
        ocr_bbox = result['ocr_bbox']
        
        # 액션 선택 (예시: 랜덤) → dict으로 포장
        action_idx = np.random.randint(0, 4)  # 0: 상, 1: 하, 2: 좌, 3: 우
        action = {
            'type': 'move',
            'direction': action_idx,
            # 만약 이동 후 위치를 계산했다면:
            # 'position': (new_x, new_y)
        }
        
        # 보상 계산 (예시: 버튼과의 거리에 기반)
        reward = 0.0
        if detected_buttons:
            # 가장 가까운 버튼 찾기
            min_dist = float('inf')
            for button in detected_buttons:
                if 'bbox' in button:
                    bbox = button['bbox']
                    button_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                    dist = np.sqrt((stamp_position[0] - button_center[0])**2 + 
                                 (stamp_position[1] - button_center[1])**2)
                    min_dist = min(min_dist, dist)
            reward = -min_dist  # 거리가 가까울수록 높은 보상
        
        # Q-값과 정책 확률 (예시)
        q_values = np.random.rand(4)  # 4개 액션에 대한 Q-값
        policy_probs = np.exp(q_values) / np.sum(np.exp(q_values))  # 소프트맥스
        
        # 스텝 로깅 (vision_state 인자는 제거)
        step_info = self.reflexion_core.process_step(
            stamp_position=stamp_position,
            detected_buttons=detected_buttons,
            action=action,
            reward=reward,
            q_values=q_values,
            policy_probs=policy_probs,
            ocr_text=ocr_text,
            ocr_bbox=ocr_bbox
        )
        
        # 호출부에서 결과를 사용하려면 반환
        return step_info

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

    def is_overlap(self, box1, box2):
        # box: (x1, y1, x2, y2)
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return x1 < x2 and y1 < y2  # 겹치면 True

    def ocr_bboxes_in_vision(self, image, bboxes):
        # image: PIL Image
        # bboxes: [(x1, y1, x2, y2), ...]
        ocr_results = []
        for bbox in bboxes:
            crop = image.crop(bbox)
            # 간단히 easyocr 사용 (실제 환경에 맞게 수정)
            reader = easyocr.Reader(['ko', 'en'], gpu=False)
            result = reader.readtext(np.array(crop))
            text = result[0][1] if result else ''
            ocr_results.append((bbox, text))
        return ocr_results

    def is_menu_in_vision(self, ocr_results, menu_queue, threshold=80):
        for bbox, text in ocr_results:
            for menu in menu_queue:
                score = fuzz.ratio(text, menu)
                if score >= threshold:
                    return True
        return False

    def log_click_event(self, step, menu, text, score, x, y, log_path="logs/click_log.csv"):
        with open(log_path, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([step, time.time(), menu, text, score, x, y])

    def visualize_cursor(self, image, position, action=None, box_color=(255,0,0), click_coords=None, status_text=None):
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

    def update_heatmap(self, position, intensity=1.0):
        if self.heatmap is None:
            self.heatmap = np.zeros((100, 100))  # 100x100 그리드로 초기화
        x, y = int(position[0] * 100), int(position[1] * 100)
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
            plt.savefig(os.path.join(self.log_dir, 'attention_heatmap.png'))
            plt.close()

        # 보상 히스토리 저장
        if self.rewards:
            plt.figure(figsize=(10, 5))
            plt.plot(self.rewards)
            plt.title('보상 변화 추이')
            plt.xlabel('스텝')
            plt.ylabel('보상')
            plt.savefig(os.path.join(self.log_dir, 'reward_history.png'))
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
            plt.savefig(os.path.join(self.log_dir, 'action_distribution.png'))
            plt.close()

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
                f.write("=== UI Agent 반성 로그 ===\n\n")
                for r in self.reflections:
                    f.write(f"스텝 {r['step']} ({r['timestamp']})\n")
                    f.write(f"목표 메뉴: {r['menu']}\n")
                    f.write(f"OCR 텍스트: {r['ocr_text']}\n")
                    f.write(f"매칭 점수: {r['score']}\n")
                    f.write(f"성공 여부: {'성공' if r['success'] else '실패'}\n")
                    f.write("-" * 50 + "\n")

    def run_episode(self, image_path: str, max_steps: int = 10) -> Dict[str, Any]:
        """
        에이전트의 한 에피소드를 실행합니다.
        """
        print(f"\n=== 에피소드 시작: {image_path} ===")
        
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
                print(f"[STEP {step+1}] 모든 메뉴를 찾았습니다. 에피소드를 종료합니다.")
                break

            # 현재 목표 메뉴 출력
            current_target = menu_queue[0]
            print(f"\n[STEP {step+1}] '{current_target}'를(을) 찾기 위해 시선을 이동중입니다...")

            action_idx = np.random.randint(0, 4)
            action = {'type': 'move', 'direction': action_idx}
            action_text = {
                0: "위로",
                1: "아래로",
                2: "왼쪽으로",
                3: "오른쪽으로"
            }[action_idx]
            print(f"  → {action_text} 이동")

            # 스탬프 위치 이동
            step_size = 0.1
            x, y = self.stamp_position
            if action_idx == 0: y = max(0.0, y - step_size)
            elif action_idx == 1: y = min(1.0, y + step_size)
            elif action_idx == 2: x = max(0.0, x - step_size)
            elif action_idx == 3: x = min(1.0, x + step_size)
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
            
            # 상태 텍스트 준비
            status_text = [
                f"Step {step+1}",
                f"목표: {current_target}",
                f"행동: {action_text} 이동"
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
                                status_text=status_text)
            
            # 보상 계산 (예시: 버튼과의 거리에 기반)
            reward = 0.0
            if result['buttons']:
                min_dist = float('inf')
                for button in result['buttons']:
                    if 'bbox' in button:
                        bbox = button['bbox']
                        button_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                        dist = np.sqrt((self.stamp_position[0] - button_center[0])**2 + 
                                     (self.stamp_position[1] - button_center[1])**2)
                        min_dist = min(min_dist, dist)
                reward = -min_dist
            
            q_values = np.random.rand(4)
            policy_probs = np.exp(q_values) / np.sum(np.exp(q_values))
            step_result = self.reflexion_core.process_step(
                stamp_position=self.stamp_position,
                detected_buttons=result['buttons'],
                action=action,
                reward=reward,
                q_values=q_values,
                policy_probs=policy_probs,
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
            'attention_heatmap_path': os.path.join(self.log_dir, 'attention_heatmap.png'),
            'reward_history_path': os.path.join(self.log_dir, 'reward_history.png'),
            'action_distribution_path': os.path.join(self.log_dir, 'action_distribution.png'),
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
    
    # OmniParserWithReflexion 인스턴스 생성
    parser = OmniParserWithReflexion(
        yolo_model=yolo_model,
        caption_model_processor=caption_model_processor,
        paddle_ocr=paddle_ocr,
        easyocr_reader=easyocr_reader
    )
    
    # 테스트 이미지 경로
    test_image = "screen2.png"
    
    # 에피소드 실행
    episode_summary = parser.run_episode(test_image)
    
    # 결과 출력
    print("\n=== 에피소드 요약 ===")
    print(f"총 보상: {episode_summary['total_reward']}")
    print(f"총 스텝: {episode_summary['steps']}")
    print(f"시각화 파일:")
    print(f"- 주의력 히트맵: {episode_summary['attention_heatmap_path']}")
    print(f"- 보상 히스토리: {episode_summary['reward_history_path']}")
    print(f"- 행동 분포: {episode_summary['action_distribution_path']}")
    print(f"- 반성 로그: {episode_summary['reflection_path']}")

if __name__ == "__main__":
    main() 