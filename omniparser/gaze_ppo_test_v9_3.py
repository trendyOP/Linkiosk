import cv2
import time
import numpy as np
import pygame
import pyautogui
import mss
import types
import difflib
import tkinter as tk
import os
import glob
import json
import sys
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
from PIL import Image
from text_normalizer import normalize_token
from typing import List
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter

# Windows DPI 스케일 문제 해결 및 인코딩 설정
try:
    import ctypes
    ctypes.windll.user32.SetProcessDPIAware()
    print("[DPI] SetProcessDPIAware enabled - pyautogui가 물리 픽셀 좌표 사용")
except Exception as e:
    print("[DPI] Failed to set DPI aware:", e)

# Windows 인코딩 문제 해결
import sys
import io
if sys.platform == "win32":
    # Windows에서 UTF-8 출력 강제
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

pyautogui.FAILSAFE = True  # Failsafe 활성화

SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ==========================
# Constants & Config
# ==========================
# 파일 경로 상수
SCREEN_PATH = "screen5.png"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gaze_ppo_v9.pt")
CLICK_SOUND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "click.wav")

# 매직 넘버들을 상수로 정의
DEFAULT_VISION_GRID_N = 32
MOUSE_MOVE_INTERVAL = 5
CLICK_DELAY = 1.143
UI_TRANSITION_DELAY = 0.05
MAX_EP_STEPS = 2000
TASK_TIMEOUT_SECONDS = 35.0  # task 타임아웃 (35초)
ALLOW_EXTRA = 2

# PaddleOCR을 가장 먼저 초기화 (torch 로드 전)
from utils.utils import get_paddle_ocr
try:
    paddle_ocr = get_paddle_ocr()
    print("[SUCCESS] PaddleOCR 초기화 완료")
    print(f"[PaddleOCR] 인스턴스 타입: {type(paddle_ocr)}")
except Exception as e:
    print(f"[WARNING] PaddleOCR 초기화 실패: {e}")
    paddle_ocr = None

# 그 다음에 torch와 다른 라이브러리들 import (지연 import)
try:
    import torch
    torch.manual_seed(SEED)
    print("[SUCCESS] torch import 완료")
except Exception as e:
    print(f"[WARNING] torch import 실패: {e}")
    torch = None

from utils.utils import check_ocr_box
try:
    from run_omniparser_with_ppo_v9 import (
        GazeKioskEnv,
        GazeActorCritic,
        VISION_GRID_N,
        build_vocab_index,
        is_inside,
        device,
        MOVE_OPTIONS,
        compute_saliency_map,
        bbox_center,
    )
    print("[SUCCESS] run_omniparser_with_ppo_v9 import 완료")
except Exception as e:
    print(f"[WARNING] run_omniparser_with_ppo_v9 import 실패: {e}")
    # 기본값 설정
    GazeKioskEnv = None
    GazeActorCritic = None
    VISION_GRID_N = DEFAULT_VISION_GRID_N
    build_vocab_index = None
    is_inside = None
    device = None
    MOVE_OPTIONS = None
    compute_saliency_map = None
    bbox_center = None

# ==========================
# Config
# ==========================
DEBUG = True
SHOW_DEBUG_WINDOW = True  # 디버그 창 표시 여부

# DPI 스케일 보정 함수
def to_screen_xy(x_raw, y_raw):
    """캡쳐된 이미지 좌표를 실제 화면 좌표로 변환"""
    try:
        # OpenCV 창의 위치와 크기 가져오기
        if cv2.getWindowProperty('OmniParser - Target Image', cv2.WND_PROP_VISIBLE) > 0:
            window_rect = cv2.getWindowImageRect('OmniParser - Target Image')
            if window_rect[2] > 0 and window_rect[3] > 0:  # width, height가 유효한 경우
                win_x, win_y, win_w, win_h = window_rect
                
                # 캡쳐된 이미지 내의 좌표를 창 좌표로 변환
                # x_raw, y_raw는 캡쳐된 이미지 내의 좌표
                screen_x = win_x + x_raw
                screen_y = win_y + y_raw
                
                if DEBUG and hasattr(to_screen_xy, '_debug_count'):
                    to_screen_xy._debug_count += 1
                else:
                    to_screen_xy._debug_count = 1
                    
                if DEBUG and to_screen_xy._debug_count % 10 == 0:  # 10번마다 디버그 출력
                    print(f"[COORD DEBUG] 캡쳐 좌표 ({x_raw:.1f}, {y_raw:.1f}) -> 화면 좌표 ({screen_x}, {screen_y})")
                    print(f"[COORD DEBUG] 창 위치: ({win_x}, {win_y}), 크기: {win_w}x{win_h}")
                
                return (int(screen_x), int(screen_y))
        
        # 창 정보를 가져올 수 없는 경우 기존 방식 사용
        print("[WARNING] 창 정보를 가져올 수 없음, 기존 좌표 변환 방식 사용")
        screen_w, screen_h = pyautogui.size()
        mon_w, mon_h = SCREEN_MONITOR["width"], SCREEN_MONITOR["height"]
        sx = screen_w / mon_w
        sy = screen_h / mon_h
        
        return (
            int(SCREEN_MONITOR["left"] * sx + x_raw * sx),
            int(SCREEN_MONITOR["top"]  * sy + y_raw * sy),
        )
        
    except Exception as e:
        print(f"[ERROR] 좌표 변환 실패: {e}, 기본값 사용")
        return (int(x_raw), int(y_raw))

# OCR 결과 시각화 함수 (디버그용)
def visualize_ocr_boxes(image, texts, boxes, output_path="ocr_debug.png"):
    """OCR 결과를 이미지에 그려서 저장 (디버그용)"""
    try:
        # 이미지 복사
        debug_img = image.copy()
        
        # 각 텍스트 박스 그리기
        for i, (text, box) in enumerate(zip(texts, boxes)):
            x1, y1, x2, y2 = box
            # 녹색 박스 그리기
            cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # 텍스트 표시
            cv2.putText(debug_img, f"{i}: {text}", (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 저장
        cv2.imwrite(output_path, debug_img)
        print(f"[DEBUG] OCR 결과 시각화 저장: {output_path}")
        return True
    except Exception as e:
        print(f"[DEBUG] OCR 시각화 실패: {e}")
        return False



# 설정 파일 읽기
def load_config():
    """config.ini 파일에서 설정을 읽어옴"""
    config = {
        'user_type': 'young',  # 기본값
        'elder_miss_click_prob': 0.4,  # elder일 때 miss_click 확률 (더 높게 조정)
        'young_miss_click_prob': 0.1,  # young일 때 miss_click 확률 (약간 높게 조정)
        'menu_queue': []
    }
    
    try:
        import configparser
        parser = configparser.ConfigParser()
        
        # config.ini 파일 경로를 동적으로 찾기
        config_paths = [
            'omniparser/config.ini',  # 기본 경로
            'config.ini',            # 현재 디렉토리
            os.path.join(os.path.dirname(__file__), 'config.ini'),  # 스크립트와 같은 디렉토리
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')  # 절대 경로
        ]
        
        config_found = False
        for config_path in config_paths:
            if os.path.exists(config_path):
                parser.read(config_path, encoding='utf-8')
                config_found = True
                print(f"[CONFIG] 설정 파일 로드: {config_path}")
                break
        
        if not config_found:
            print(f"[WARNING] config.ini 파일을 찾을 수 없습니다. 기본값을 사용합니다.")
            print(f"[WARNING] 시도한 경로들: {config_paths}")
        
        if 'User' in parser:
            config['user_type'] = parser.get('User', 'type', fallback='young')
        
        if 'Menu' in parser:
            queue_str = parser.get('Menu', 'queue', fallback='')
            if queue_str:
                config['menu_queue'] = [item.strip() for item in queue_str.split(',')]
        
        print(f"[CONFIG] User type: {config['user_type']}")
        print(f"[CONFIG] Menu queue: {config['menu_queue']}")
        print(f"[CONFIG] Elder miss_click probability: {config['elder_miss_click_prob']}")
        print(f"[CONFIG] Young miss_click probability: {config['young_miss_click_prob']}")
        
    except Exception as e:
        print(f"[WARNING] 설정 파일 읽기 실패: {e}")
    
    return config

# 설정 로드
CONFIG = load_config()

# 화면 설정 (자동 감지)
def get_screen_monitor():
    """화면 해상도를 자동으로 감지"""
    try:
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # 주 모니터
            return {
                "top": monitor["top"],
                "left": monitor["left"], 
                "width": monitor["width"],
                "height": monitor["height"]
            }
    except Exception as e:
        print(f"[WARNING] 화면 감지 실패, 기본값 사용: {e}")
        return {"top": 0, "left": 0, "width": 1920, "height": 1080}

SCREEN_MONITOR = get_screen_monitor()

# 테스트 설정 - config에서 읽어온 queue 사용
if CONFIG['menu_queue']:
    TEST_TASKS = [CONFIG['menu_queue']]
else:
    TEST_TASKS = [["매장식사","아메리카노","주문담기","더담기","스무디","수박 주스", "주문담기", "더담기","베이커리","햄&치즈 샌드위치","주문담기", "결제하기", "확인","신용카드","대기","예"]]

# ==========================
# Image Processing (Simplified)
# ==========================

# ==========================
# Simplified OCR Processing
# ==========================

# ==========================
# Helpers
# ==========================
def _set_goal_sequence(self, seq: List[str]):
    self._goal_seq = seq
    self.goal_idx  = 0

def to_xyxy(bb):
    arr = np.array(bb, dtype=float).reshape(-1, 2)
    x1, y1 = arr.min(axis=0)
    x2, y2 = arr.max(axis=0)
    return [x1, y1, x2, y2]

def _bbox_center(bb):
    x1,y1,x2,y2 = to_xyxy(bb)
    return (x1+x2)/2, (y1+y2)/2

def _bbox_wh(bb):
    x1,y1,x2,y2 = to_xyxy(bb)
    w = max(1, x2-x1); h = max(1, y2-y1)
    return w, h, w*h

def _prefix_with_price(tgt_norm, txt_norm):
    import re
    pat = rf'^{re.escape(tgt_norm)}[0-9\s원]+$'
    return re.match(pat, txt_norm) is not None

def _has_price(txt_n):
    return any(c.isdigit() for c in txt_n) or '원' in txt_n

def _is_word_boundary_match(tgt, txt):
    import re
    pat = rf'(^|[^가-힣A-Za-z0-9]){re.escape(tgt)}([^가-힣A-Za-z0-9]|$)'
    return re.search(pat, txt) is not None

def _better_match(tgt_norm, txt_norm):
    if len(txt_norm) - len(tgt_norm) > ALLOW_EXTRA:
        return False
    ratio = difflib.SequenceMatcher(None, tgt_norm, txt_norm).ratio()
    return ratio >= 0.9

def _is_ocr_error_match(tgt_norm, txt_norm):
    """OCR 오류로 인한 잘못된 인식을 처리합니다."""
    # HOT -> Hoi 같은 오류 패턴
    ocr_error_patterns = {
        'hot': ['hoi', 'hot', 'h0t', 'h0i'],
        '햄&치즈': ['햄&치즈', '햄앤치즈', '햄치즈', '햄치즈', '햄&치즈샌드위치'],
        '샌드위치': ['샌드위치', '샌드위츠', '샌드위치'],
        '햄&치즈샌드위치': ['햄&치즈샌드위치', '햄치즈샌드위치', '햄앤치즈샌드위치'],
        '신용카드': ['신용카드', '신용가드', '신용카드결제', '신용가드결제'],
        '더담기': ['더담기', '담기', '더 담기'],
        '더 담기': ['더담기', '담기', '더 담기'],
    }
    
    tgt_lower = tgt_norm.lower()
    txt_lower = txt_norm.lower()
    
    for pattern, variations in ocr_error_patterns.items():
        if tgt_lower == pattern and txt_lower in variations:
            return True
    
    return False

def _is_split_text_match(tgt_norm, all_texts):
    """분리된 텍스트가 목표와 매칭되는지 확인합니다."""
    # "햄&치즈 샌드위치" -> ["햄&치즈", "샌드위치"] 같은 경우
    # normalize_token이 &를 제거하므로 정규화된 형태로 매칭
    split_patterns = {
        '햄치즈샌드위치': ['햄치즈', '샌드위치'],
        '햄치즈 샌드위치': ['햄치즈', '샌드위치'],
        'hamcheesesandwich': ['hamcheese', 'sandwich'],
        'hamcheese sandwich': ['hamcheese', 'sandwich'],
        '햄&치즈샌드위치': ['햄&치즈', '샌드위치'],
        '햄&치즈 샌드위치': ['햄&치즈', '샌드위치'],
        '더담기': ['더', '담기'],
        '더 담기': ['더', '담기'],
    }
    
    all_texts_norm = [normalize_token(t) for t in all_texts]
    
    for pattern, parts in split_patterns.items():
        if tgt_norm == pattern:
            # 모든 부분이 존재하는지 확인 (정규화된 형태로)
            if all(part in all_texts_norm for part in parts):
                return True
    
    return False

def _goal_step_postprocess(self, in_view):
    found = False
    self._last_candidates = []
    if self.goal_idx >= len(self._goal_seq):
        return False

    tgt_raw = self._goal_seq[self.goal_idx]
    tgt_norm = normalize_token(tgt_raw)

    # 특별한 goal 처리: "대기"
    if tgt_raw == "대기":
        print(f"[WAIT] 5초 대기 실행")
        time.sleep(5.0)  # 5초 대기
        self.goal_idx += 1
        return True

    # 매칭 품질 점수 계산
    candidates_with_score = []
    
    # 분리된 텍스트 매칭 확인
    if _is_split_text_match(tgt_norm, self.ocr_txt):
        if DEBUG:
            print(f"[SPLIT MATCH] '{tgt_raw}' -> 분리된 텍스트로 매칭됨")
        # 분리된 텍스트의 모든 부분을 후보로 추가
        split_patterns = {
            '햄치즈샌드위치': ['햄치즈', '샌드위치'],
            '햄치즈 샌드위치': ['햄치즈', '샌드위치'],
            'hamcheesesandwich': ['hamcheese', 'sandwich'],
            'hamcheese sandwich': ['hamcheese', 'sandwich'],
            '햄&치즈샌드위치': ['햄&치즈', '샌드위치'],
            '햄&치즈 샌드위치': ['햄&치즈', '샌드위치'],
        }
        
        for pattern, parts in split_patterns.items():
            if tgt_norm == pattern:
                # 분리된 텍스트의 모든 부분을 찾기
                found_parts = []
                for part in parts:
                    for txt, bb in zip(self.ocr_txt, self.ocr_bb):
                        if not txt or not is_inside(bb, self._vbox()):
                            continue
                        txt_norm = normalize_token(txt)
                        if txt_norm == part:
                            found_parts.append((txt, bb))
                            break
                
                # 모든 부분이 찾아졌으면 통합된 박스 생성
                if len(found_parts) == len(parts):
                    # 모든 박스의 중심점을 평균내어 통합된 박스 생성
                    all_centers = []
                    for txt, bb in found_parts:
                        center_x, center_y = _bbox_center(bb)
                        all_centers.append((center_x, center_y))
                    
                    # 중심점들의 평균 계산
                    avg_center_x = sum(cx for cx, cy in all_centers) / len(all_centers)
                    avg_center_y = sum(cy for cx, cy in all_centers) / len(all_centers)
                    
                    # 통합된 박스 생성 (중심점 기준으로 적당한 크기)
                    box_size = 100  # 적당한 클릭 영역 크기
                    integrated_bbox = [
                        [avg_center_x - box_size/2, avg_center_y - box_size/2],
                        [avg_center_x + box_size/2, avg_center_y - box_size/2],
                        [avg_center_x + box_size/2, avg_center_y + box_size/2],
                        [avg_center_x - box_size/2, avg_center_y + box_size/2]
                    ]
                    
                    # 통합된 텍스트 생성
                    integrated_text = " ".join([txt for txt, bb in found_parts])
                    
                    candidates_with_score.append({
                        'text': integrated_text,
                        'bbox': integrated_bbox,
                        'score': 85,  # 분리된 텍스트 매칭 점수
                        'match_type': 'split_integrated',
                        'len_diff': 0,
                        'original_score': 85
                    })
                    
                    if DEBUG:
                        print(f"[SPLIT INTEGRATED] '{integrated_text}' at center ({avg_center_x:.1f}, {avg_center_y:.1f})")
                        print(f"  - Found parts: {[txt for txt, bb in found_parts]}")
                        print(f"  - Integrated bbox: {integrated_bbox}")
                break
    
    # 개별 텍스트 매칭
    for txt, bb in zip(self.ocr_txt, self.ocr_bb):
        if not txt or not is_inside(bb, self._vbox()):
            continue
        
        txt_norm = normalize_token(txt)
        score = 0
        match_type = "none"
        
        # 1. 정확 매칭: 최고 점수 (100)
        if tgt_norm == txt_norm:
            score = 100
            match_type = "exact"
        # 2. OCR 오류 매칭: 높은 점수 (90)
        elif _is_ocr_error_match(tgt_norm, txt_norm):
            score = 90
            match_type = "ocr_error"
        # 3. 단어 경계 매칭: 높은 점수 (80)
        elif _is_word_boundary_match(tgt_norm, txt_norm):
            score = 80
            match_type = "boundary"
        # 4. 가격 포함 매칭: 중간 점수 (70)
        elif _prefix_with_price(tgt_norm, txt_norm):
            score = 70
            match_type = "price"
        # 5. fuzzy 매칭: 낮은 점수 (60)
        elif tgt_norm in txt_norm and _better_match(tgt_norm, txt_norm):
            score = 60
            match_type = "fuzzy"
        # 6. 부분 포함 매칭: 낮은 점수 (50) - 완화된 매칭
        elif tgt_norm in txt_norm or txt_norm in tgt_norm:
            # 길이 차이가 너무 크지 않으면 매칭
            len_diff = abs(len(txt_norm) - len(tgt_norm))
            if len_diff <= 3:  # 3글자 이내 차이
                score = 50
                match_type = "partial_contain"
        # 7. 유사도 매칭: 매우 낮은 점수 (30) - SequenceMatcher 사용
        elif difflib.SequenceMatcher(None, tgt_norm, txt_norm).ratio() >= 0.7:
            score = 30
            match_type = "similarity"
        # 8. 부분 매칭: 매우 낮은 점수 (40) - 햄&치즈 샌드위치 특별 처리
        elif tgt_norm == '햄&치즈샌드위치' and ('햄' in txt_norm or '치즈' in txt_norm or '샌드위치' in txt_norm):
            score = 40
            match_type = "partial_ham_cheese"
        
        # 점수가 있는 경우 추가 정보와 함께 저장
        if score > 0:
            # 길이 차이 보정 (짧을수록 좋음)
            len_diff = abs(len(txt_norm) - len(tgt_norm))
            len_bonus = max(0, 10 - len_diff)  # 길이 차이가 적을수록 보너스
            
            # 최종 점수 계산
            final_score = score + len_bonus
            
            candidates_with_score.append({
                'text': txt,
                'bbox': bb,
                'score': final_score,
                'match_type': match_type,
                'len_diff': len_diff,
                'original_score': score
            })
    
    # 점수 순으로 정렬하여 최고 점수 선택
    if candidates_with_score:
        candidates_with_score.sort(key=lambda x: x['score'], reverse=True)
        best_score = candidates_with_score[0]['score']
        
        # 최고 점수와 같은 모든 후보 선택
        best_candidates = [c for c in candidates_with_score if c['score'] == best_score]
        
        # 디버그 정보 출력
        if DEBUG:
            print(f"[GOAL MATCH] '{tgt_raw}' -> {len(best_candidates)} candidates:")
            for i, c in enumerate(best_candidates[:3]):  # 상위 3개만 출력
                print(f"  {i+1}. '{c['text']}' (score: {c['score']}, type: {c['match_type']}, len_diff: {c['len_diff']})")
            
            # 햄&치즈 샌드위치 특별 디버그
            if tgt_raw == "햄&치즈샌드위치":
                print(f"[HAM_CHEESE DEBUG] Target: '{tgt_raw}', Normalized: '{tgt_norm}'")
                print(f"[HAM_CHEESE DEBUG] Available OCR texts: {self.ocr_txt}")
                print(f"[HAM_CHEESE DEBUG] All candidates: {candidates_with_score}")
        
        # 강한 매칭만 골라냄
        STRONG_TYPES = {'exact', 'ocr_error', 'boundary', 'price'}
        strong = [c for c in candidates_with_score if c['match_type'] in STRONG_TYPES]

        if strong:
            # 강한 후보들만 클릭 후보로
            strong.sort(key=lambda x: x['score'], reverse=True)
            self._last_candidates = [(c['text'], c['bbox']) for c in strong if is_inside(c['bbox'], self._vbox())]
            self.goal_idx += 1
            found = True
        else:
            # 강한 매칭이 없으면 '발견 안 됨' 처리 (부분/유사만 있을 때는 탐색만 지속)
            self._last_candidates = []
            found = False
    
    return found

def _pick_click_candidate(cands, tgt_norm):
    if not cands:
        return None
    
    def calculate_score(item):
        txt, bb = item
        txt_n = normalize_token(txt)
        
        # 0. 텍스트 매칭 점수 (가장 중요, 최대 50점)
        if txt_n == tgt_norm:
            match_score = 50  # 정확 매칭
        elif _is_word_boundary_match(tgt_norm, txt_n):
            match_score = 45  # 단어 경계 매칭
        elif _prefix_with_price(tgt_norm, txt_n):
            match_score = 40  # 가격 포함 매칭
        elif tgt_norm in txt_n and _better_match(tgt_norm, txt_n):
            match_score = 35  # fuzzy 매칭
        else:
            match_score = 0   # 매칭 안됨
        
        # 1. 길이 차이 점수 (짧을수록 좋음, 최대 15점)
        len_diff = abs(len(txt_n) - len(tgt_norm))
        len_score = max(0, 15 - len_diff * 1.5)
        
        # 2. 화면 중앙에서의 거리 점수 (가까울수록 좋음, 최대 20점)
        cx, cy = _bbox_center(bb)
        # 화면 크기를 동적으로 가져오기 위해 기본값 사용
        screen_center_x, screen_center_y = 960, 540  # 1920x1080 기준
        center_dist = math.hypot(cx - screen_center_x, cy - screen_center_y)
        # 거리가 500px 이내면 최대 점수, 그 이상이면 감소
        center_score = max(0, 20 - (center_dist / 500) * 20)
        
        # 3. 영역 크기 점수 (적당한 크기일수록 좋음, 최대 10점)
        w, h, area = _bbox_wh(bb)
        # 500~2000px² 범위가 이상적
        if 500 <= area <= 2000:
            size_score = 10
        elif area < 500:
            size_score = 10 * (area / 500)  # 작을수록 감소
        else:
            size_score = max(0, 10 * (4000 / area))  # 클수록 감소
        
        # 4. 텍스트 품질 점수 (숫자나 특수문자가 적을수록 좋음, 최대 5점)
        # 숫자 개수
        digit_count = sum(1 for c in txt_n if c.isdigit())
        # 특수문자 개수 (한글, 영문, 숫자 제외)
        special_count = sum(1 for c in txt_n if not (c.isalnum() or '\u3131' <= c <= '\u318E'))
        quality_score = max(0, 5 - (digit_count + special_count) * 1)
        
        # 총점 계산 (텍스트 매칭이 가장 중요)
        total_score = match_score + len_score + center_score + size_score + quality_score
        
        return {
            'item': item,
            'total_score': total_score,
            'match_score': match_score,
            'len_score': len_score,
            'center_score': center_score,
            'size_score': size_score,
            'quality_score': quality_score,
            'len_diff': len_diff,
            'center_dist': center_dist,
            'area': area,
            'digit_count': digit_count,
            'special_count': special_count
        }
    
    # 모든 후보에 대해 점수 계산
    scored_candidates = [calculate_score(cand) for cand in cands]
    
    # 총점 순으로 정렬
    scored_candidates.sort(key=lambda x: x['total_score'], reverse=True)
    
    # 디버그 정보 출력
    if DEBUG and scored_candidates:
        best = scored_candidates[0]
        txt, bb = best['item']
        print(f"[CLICK SELECT] '{txt}' (total: {best['total_score']:.1f})")
        print(f"  - match: {best.get('match_score', 0):.1f}, len: {best['len_score']:.1f}, center: {best['center_score']:.1f}, size: {best['size_score']:.1f}, quality: {best['quality_score']:.1f}")
        print(f"  - len_diff: {best['len_diff']}, center_dist: {best['center_dist']:.1f}, area: {best['area']:.1f}")
    
    # 최고 점수 후보 반환
    return scored_candidates[0]['item'] if scored_candidates else None

def biased_sample(net, obs_vec, env, alpha=2.0):
    with torch.no_grad():
        logits, _ = net(torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0))
        logits = logits.squeeze(0)  # shape [A]
    
    hint_bias = []
    for (dx,dy) in MOVE_OPTIONS:
        gx = int(np.clip(env.gx+dx,0,env.N-1))
        gy = int(np.clip(env.gy+dy,0,env.N-1))
        
        # 기본 힌트 맵만 사용 (단순화)
        base_hint = env.hint_map[gy,gx]
        hint_bias.append(base_hint)
    
    hint_bias = torch.tensor(hint_bias, dtype=torch.float32, device=device)
    biased_logits = logits + alpha * hint_bias
    return torch.distributions.Categorical(logits=biased_logits).sample().item()

# ─── Simplified Processing ───────────────────────────────────────

# 이동방향 → action id 매핑
# MOVE_OPTIONS가 None인 경우 기본값 설정
if MOVE_OPTIONS is None:
    MOVE_OPTIONS = ['up', 'down', 'left', 'right', 'click']
MOVE2IDX = {d: i for i, d in enumerate(MOVE_OPTIONS)}

def plan_move(curr, tgt):
    dx = 0 if curr[0] == tgt[0] else (1 if tgt[0] > curr[0] else -1)
    dy = 0 if curr[1] == tgt[1] else (1 if tgt[1] > curr[1] else -1)
    if (dx,dy) in MOVE2IDX:
        return MOVE2IDX[(dx,dy)]
    best, best_d = 0, 1e9
    for (mx,my), idx in MOVE2IDX.items():
        d = abs(mx-dx)+abs(my-dy)
        if d < best_d:
            best, best_d = idx, d
    print(f"[plan_move] unmapped ({dx},{dy}) -> use {best}")
    return best

# ==========================
# Image Display Functions
# ==========================
def _display_image_on_screen(image, image_path):
    """이미지를 전체 화면에 표시"""
    try:
        # 화면 크기 가져오기
        screen_width, screen_height = pyautogui.size()
        
        # 이미지 크기
        img_width, img_height = image.size
        
        # 화면에 맞게 이미지 크기 조정 (비율 유지)
        scale_x = screen_width / img_width
        scale_y = screen_height / img_height
        scale = min(scale_x, scale_y)  # 비율 유지하면서 화면에 맞춤
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # 이미지 리사이즈
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 중앙에 배치하기 위한 좌표 계산
        x_offset = (screen_width - new_width) // 2
        y_offset = (screen_height - new_height) // 2
        
        # 새로운 이미지 생성 (검은 배경)
        display_image = Image.new('RGB', (screen_width, screen_height), (0, 0, 0))
        display_image.paste(resized_image, (x_offset, y_offset))
        
        # OpenCV로 변환
        display_cv = cv2.cvtColor(np.array(display_image), cv2.COLOR_RGB2BGR)
        
        # 전체 화면 창 생성
        cv2.namedWindow('OmniParser - Target Image', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('OmniParser - Target Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # 이미지 표시
        cv2.imshow('OmniParser - Target Image', display_cv)
        
        print(f"[DISPLAY] 이미지 표시 완료: {image_path}")
        print(f"[DISPLAY] 원본 크기: {img_width}x{img_height}")
        print(f"[DISPLAY] 표시 크기: {new_width}x{new_height}")
        print(f"[DISPLAY] 화면 크기: {screen_width}x{screen_height}")
        print(f"[DISPLAY] 스케일: {scale:.2f}")
        
        # 잠시 대기 (이미지가 표시되도록)
        cv2.waitKey(100)
        
    except Exception as e:
        print(f"[ERROR] 이미지 표시 실패: {e}")

def _close_image_display():
    """이미지 표시 창 닫기"""
    try:
        cv2.destroyWindow('OmniParser - Target Image')
        print("[DISPLAY] 이미지 표시 창 닫기 완료")
    except:
        pass

def _capture_displayed_image():
    """화면에 표시된 이미지 창을 캡쳐"""
    try:
        # OpenCV 창이 활성화되어 있는지 확인
        if cv2.getWindowProperty('OmniParser - Target Image', cv2.WND_PROP_VISIBLE) > 0:
            # 창의 위치와 크기 가져오기
            window_rect = cv2.getWindowImageRect('OmniParser - Target Image')
            if window_rect[2] > 0 and window_rect[3] > 0:  # width, height가 유효한 경우
                x, y, w, h = window_rect
                
                # 화면 캡쳐
                with mss.mss() as sct:
                    # 창 영역 캡쳐
                    monitor = {
                        "top": y,
                        "left": x, 
                        "width": w,
                        "height": h
                    }
                    screenshot = sct.grab(monitor)
                    
                    # PIL Image로 변환
                    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                    
                    print(f"[CAPTURE] 화면 캡쳐 완료: {w}x{h} at ({x}, {y})")
                    return img
        else:
            print("[CAPTURE] 이미지 창이 표시되지 않음")
            return None
            
    except Exception as e:
        print(f"[CAPTURE] 화면 캡쳐 실패: {e}")
        return None

# ==========================
# Image OCR Processing
# ==========================
def _process_image_ocr(self: GazeKioskEnv, image_path: str):
    """업로드된 이미지를 화면에 표시하고, 표시된 화면을 캡쳐해서 OCR 처리"""
    t0 = time.perf_counter()
    
    if DEBUG:
        print(f"[OCR PROCESS] Processing image: {image_path}")
    
    # 원본 이미지 로드 (화면 표시용)
    original_image = Image.open(image_path)
    
    # 화면에 이미지 표시
    _display_image_on_screen(original_image, image_path)
    
    # 잠시 대기 (이미지가 완전히 표시되도록)
    time.sleep(0.5)
    
    # 화면에 표시된 이미지 캡쳐
    captured_image = _capture_displayed_image()
    
    if captured_image is None:
        print("[WARNING] 화면 캡쳐 실패, 원본 이미지 사용")
        self.image = original_image
    else:
        print("[SUCCESS] 화면 캡쳐 성공, 캡쳐된 이미지로 OCR 수행")
        self.image = captured_image
        
        # 디버그용으로 캡쳐된 이미지 저장
        if DEBUG:
            captured_image.save("captured_screen.png")
            print("[DEBUG] 캡쳐된 이미지를 captured_screen.png로 저장")
    
    # 캡쳐된 이미지의 크기 설정
    self.W, self.H = self.image.size
    self.cell_w, self.cell_h = self.W / self.N, self.H / self.N

    if DEBUG:
        print(f"[IMAGE INFO] OCR 대상 이미지 크기: {self.W}x{self.H}")
        print(f"[IMAGE INFO] 셀 크기: {self.cell_w:.1f}x{self.cell_h:.1f}")

    # 캡쳐된 이미지로 OCR 실행
    self.ocr_txt, self.ocr_bb = check_ocr_box(
        self.image,
        display_img=False,
        output_bb_format="xyxy",
        use_paddleocr=True,
        ocr_engine='paddleocr',
        roi=(0.0, 1.0, 0.0, 1.0),
        paddle_det_limit_side_len=1920,
        paddle_rec_score_thresh=0.35,
        paddle_lang='korean'
    )

    # Saliency map 계산 (캡쳐된 이미지 기준)
    self.saliency_map = compute_saliency_map(self.image, self.N)
    self.token_cells.fill(False)
    for bb in self.ocr_bb:
        x1, y1, x2, y2 = to_xyxy(bb)
        x, y = (x1+x2)/2, (y1+y2)/2
        gx = min(int(x / self.cell_w), self.N - 1)
        gy = min(int(y / self.cell_h), self.N - 1)
        self.token_cells[gy, gx] = True

    self.hint_map = (self.token_cells.astype(np.float32) * 0.7 +
                     self.saliency_map.astype(np.float32) * 0.3)
    self.total_token_cells = self.token_cells.sum()

    if DEBUG:
        print(f"[OCR COMPLETE] {time.perf_counter()-t0:.3f}s - Found {len(self.ocr_txt)} texts")
        for i, txt in enumerate(self.ocr_txt):
            print(f"  {i+1:2d}. '{txt}'")
    
    return self._obs()

# ==========================
# Debug Window Functions (Tkinter 기반)
# ==========================
# (imports consolidated; ttk 미사용)
from PIL import ImageTk

class DebugWindow:
    def __init__(self):
        if not SHOW_DEBUG_WINDOW:
            return
            
        self.root = tk.Tk()
        self.root.title("OmniParser Debug")
        self.root.geometry("500x400")  # 크기 증가
        self.root.attributes('-topmost', True)  # 항상 최상위
        self.root.resizable(False, False)
        
        # 다크 테마 설정
        self.root.configure(bg='#2b2b2b')  # 어두운 배경색
        
        # 화면 크기 가져오기
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 창 크기
        window_width = 500
        window_height = 400
        
        # 오른쪽 아래 위치 계산
        x = screen_width - window_width - 20  # 오른쪽에서 20px 여백
        y = screen_height - window_height - 40  # 아래에서 40px 여백 (작업표시줄 고려)
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 프레임 생성 (다크 테마)
        self.frame = tk.Frame(self.root, bg='#2b2b2b', padx=10, pady=10)
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 정보 라벨들 (다크 테마)
        label_style = {'bg': '#2b2b2b', 'fg': '#ffffff', 'font': ('Arial', 11)}
        
        self.time_label = tk.Label(self.frame, text="Time: 0.0s", **label_style)
        self.time_label.grid(row=0, column=0, sticky=tk.W, pady=3)
        
        self.goal_label = tk.Label(self.frame, text="Goal: None", **label_style)
        self.goal_label.grid(row=1, column=0, sticky=tk.W, pady=3)
        
        self.gaze_label = tk.Label(self.frame, text="Gaze: (0, 0)", **label_style)
        self.gaze_label.grid(row=2, column=0, sticky=tk.W, pady=3)
        
        # 클릭 수 정보 라벨
        self.clicks_label = tk.Label(self.frame, text="Clicks: 0", **label_style)
        self.clicks_label.grid(row=3, column=0, sticky=tk.W, pady=3)
        
        # 상태 표시 (다크 테마)
        self.status_label = tk.Label(self.frame, text="Status: Ready", 
                                   bg='#2b2b2b', fg='#00ff00', font=('Arial', 12, 'bold'))
        self.status_label.grid(row=4, column=0, sticky=tk.W, pady=12)
        
        # 로그 텍스트 영역 (다크 테마)
        self.log_text = tk.Text(self.frame, height=10, width=60, 
                               font=('Consolas', 9), bg='#1e1e1e', fg='#ffffff',
                               insertbackground='#ffffff', selectbackground='#404040')
        self.log_text.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # 스크롤바 (다크 테마)
        scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.log_text.yview,
                               bg='#404040', troughcolor='#2b2b2b')
        scrollbar.grid(row=5, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # 종료 버튼 (다크 테마)
        self.close_button = tk.Button(self.frame, text="Close", command=self.root.destroy,
                                    bg='#404040', fg='#ffffff', font=('Arial', 10),
                                    relief='flat', padx=20, pady=5)
        self.close_button.grid(row=6, column=0, pady=8)
        
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        
    def update_info(self, step, goal_tok, env, start_time, total_clicks):
        if not SHOW_DEBUG_WINDOW or not hasattr(self, 'root'):
            return
            
        try:
            elapsed_time = time.time() - start_time
            self.time_label.config(text=f"Time: {elapsed_time:.1f}s")
            self.goal_label.config(text=f"Goal: {goal_tok or 'None'}")
            self.gaze_label.config(text=f"Gaze: ({env.gx}, {env.gy})")
            self.clicks_label.config(text=f"Clicks: {total_clicks}")
            
            # 상태 업데이트 (다크 테마)
            self.status_label.config(text="Status: Processing", fg="#00ff00")
                
            self.root.update()
        except:
            pass  # 창이 닫혔을 때 예외 무시
    
    def add_log(self, message):
        if not SHOW_DEBUG_WINDOW or not hasattr(self, 'root'):
            return
            
        try:
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)  # 자동 스크롤
            self.root.update()
        except:
            pass
    
    def destroy(self):
        if hasattr(self, 'root'):
            try:
                self.root.destroy()
            except:
                pass

# 전역 디버그 창 인스턴스
debug_window = None

def create_debug_window(env, goal_tok, step, start_time, total_clicks):
    """디버그 창 생성 및 업데이트"""
    global debug_window
    
    if not SHOW_DEBUG_WINDOW:
        return
    
    # 첫 실행시 창 생성
    if debug_window is None:
        debug_window = DebugWindow()
    
    # 정보 업데이트
    debug_window.update_info(step, goal_tok, env, start_time, total_clicks)

# ==========================
# Task Selection Functions
# ==========================

def calculate_text_priority(ocr_txt, ocr_bb, image_width, image_height):
    """텍스트의 우선순위를 계산 (테두리 쪽이 높은 우선순위)"""
    priorities = []
    
    for i, (text, bbox) in enumerate(zip(ocr_txt, ocr_bb)):
        if not text or len(text.strip()) < 2:  # 너무 짧은 텍스트 제외
            priorities.append(0)
            continue
            
        # bbox 중심점 계산
        x1, y1, x2, y2 = to_xyxy(bbox)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 화면 중앙에서의 거리 계산
        screen_center_x = image_width / 2
        screen_center_y = image_height / 2
        distance_from_center = math.sqrt((center_x - screen_center_x)**2 + (center_y - screen_center_y)**2)
        
        # 화면 대각선 길이의 절반을 기준으로 정규화 (0~1)
        max_distance = math.sqrt(image_width**2 + image_height**2) / 2
        normalized_distance = min(distance_from_center / max_distance, 1.0)
        
        # 테두리 쪽일수록 높은 점수 (0~100)
        edge_priority = normalized_distance * 100
        
        # 텍스트 길이 보너스 (적당한 길이의 텍스트 선호)
        text_length = len(text.strip())
        if 3 <= text_length <= 10:
            length_bonus = 10
        elif text_length > 10:
            length_bonus = 5
        else:
            length_bonus = 0
            
        # 특수문자나 숫자가 적을수록 보너스
        special_chars = sum(1 for c in text if not (c.isalnum() or '\u3131' <= c <= '\u318E'))
        digit_chars = sum(1 for c in text if c.isdigit())
        quality_bonus = max(0, 10 - (special_chars + digit_chars))
        
        # 최종 우선순위 점수
        total_priority = edge_priority + length_bonus + quality_bonus
        priorities.append(total_priority)
    
    return priorities

def select_random_tasks(ocr_txt, ocr_bb, image_width, image_height, num_tasks=5):
    """OCR 텍스트에서 테두리 우선으로 랜덤 5개 task 선택 (숫자 포함 텍스트 제외)"""
    if len(ocr_txt) == 0:
        print("[WARNING] OCR 텍스트가 없습니다.")
        return []
    
    # 숫자가 포함된 텍스트 필터링
    def has_numbers(text):
        """텍스트에 숫자가 포함되어 있는지 확인"""
        return any(c.isdigit() for c in text)
    
    # 숫자가 포함되지 않은 텍스트만 필터링
    filtered_pairs = []
    for i, (text, bbox) in enumerate(zip(ocr_txt, ocr_bb)):
        if not has_numbers(text.strip()):
            filtered_pairs.append((text, bbox, i))
        else:
            print(f"[TASK FILTER] 숫자 포함으로 제외: '{text}'")
    
    if len(filtered_pairs) == 0:
        print("[WARNING] 숫자가 포함되지 않은 텍스트가 없습니다.")
        return []
    
    print(f"[TASK FILTER] {len(ocr_txt)}개 텍스트 중 {len(filtered_pairs)}개가 숫자 미포함")
    
    # 필터링된 텍스트에 대해 우선순위 계산
    filtered_texts = [pair[0] for pair in filtered_pairs]
    filtered_bboxes = [pair[1] for pair in filtered_pairs]
    priorities = calculate_text_priority(filtered_texts, filtered_bboxes, image_width, image_height)
    
    # 텍스트와 우선순위를 함께 정렬
    text_priority_pairs = list(zip(filtered_texts, filtered_bboxes, priorities))
    text_priority_pairs.sort(key=lambda x: x[2], reverse=True)  # 우선순위 높은 순으로 정렬
    
    # 상위 70%에서 랜덤 선택 (테두리 우선이지만 완전히 랜덤도 허용)
    top_count = max(1, int(len(text_priority_pairs) * 0.7))
    top_candidates = text_priority_pairs[:top_count]
    
    # 실제 선택할 개수 (전체 텍스트 수와 요청 개수 중 작은 값)
    actual_num_tasks = min(num_tasks, len(top_candidates))
    
    # 랜덤 선택
    selected_tasks = random.sample(top_candidates, actual_num_tasks)
    
    # 결과 정리
    selected_texts = []
    for text, bbox, priority in selected_tasks:
        selected_texts.append({
            'text': text.strip(),
            'bbox': bbox,
            'priority': priority
        })
    
    print(f"[TASK SELECTION] {len(filtered_pairs)}개 숫자 미포함 텍스트 중 {len(selected_texts)}개 task 선택:")
    for i, task in enumerate(selected_texts, 1):
        print(f"  {i}. '{task['text']}' (우선순위: {task['priority']:.1f})")
    
    return selected_texts

def calculate_task_averages(task_results):
    """5개 task 결과의 평균을 계산"""
    if not task_results:
        return None
    
    valid_results = [r for r in task_results if r is not None]
    if not valid_results:
        return None
    
    # 평균 계산
    avg_result = {
        'total_tasks': len(task_results),
        'successful_tasks': sum(1 for r in valid_results if r.get('success', False)),
        'success_rate': sum(1 for r in valid_results if r.get('success', False)) / len(valid_results) * 100,
        'avg_steps': sum(r.get('steps', 0) for r in valid_results) / len(valid_results),
        'avg_clicks': sum(r.get('clicks', 0) for r in valid_results) / len(valid_results),
        'avg_duration': sum(r.get('duration', 0.0) for r in valid_results) / len(valid_results),
        'total_duration': sum(r.get('duration', 0.0) for r in valid_results),
        'individual_results': task_results
    }
    
    return avg_result

# ==========================
# Gaze Heatmap Functions
# ==========================

def create_gaze_heatmap(mouse_history, image_size, output_path, title="Gaze Heatmap"):
    """시선 집중도 heatmap 생성"""
    try:
        if not mouse_history:
            print("[HEATMAP] 마우스 히스토리가 없습니다.")
            return None
            
        # 이미지 크기
        img_width, img_height = image_size
        
        # 히스토그램 생성 (시선 집중도)
        heatmap = np.zeros((img_height, img_width), dtype=np.float32)
        
        # 각 마우스 위치에 가중치 추가
        for x, y, duration, event_type in mouse_history:
            # 좌표가 이미지 범위 내에 있는지 확인
            if 0 <= x < img_width and 0 <= y < img_height:
                # 이벤트 타입별 가중치
                if event_type == 'click':
                    weight = 3.0  # 클릭은 높은 가중치
                elif event_type == 'hover':
                    weight = duration * 2.0  # 호버 시간에 비례
                else:
                    weight = 1.0  # 일반 이동
                
                # 가우시안 분포로 주변 픽셀에도 영향
                sigma = 20  # 분산 크기
                y_coords, x_coords = np.ogrid[:img_height, :img_width]
                
                # 거리 계산
                dist_sq = (x_coords - x)**2 + (y_coords - y)**2
                
                # 가우시안 가중치 적용
                gaussian_weight = np.exp(-dist_sq / (2 * sigma**2))
                heatmap += gaussian_weight * weight
        
        # 정규화 (0-1 범위)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # 가우시안 필터로 부드럽게
        heatmap = gaussian_filter(heatmap, sigma=5)
        
        # 시각화
        plt.figure(figsize=(12, 8))
        # extent를 사용해서 좌표계를 명시적으로 설정 (좌우반전 방지)
        plt.imshow(heatmap, cmap='hot', alpha=0.7, interpolation='bilinear', 
                   extent=[0, img_width, img_height, 0], origin='upper')
        plt.colorbar(label='Gaze Intensity')
        plt.title(f'{title}\n(Red=High, Blue=Low)')
        plt.axis('off')
        
        # 클릭 위치를 빨간 점으로 표시 (좌표 검증용)
        click_positions = [(x, y) for x, y, duration, event_type in mouse_history if event_type == 'click']
        for x, y in click_positions:
            plt.plot(x, y, 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
        
        # 통계 정보 추가
        total_points = len(mouse_history)
        clicks = len([h for h in mouse_history if h[3] == 'click'])
        avg_intensity = np.mean(heatmap)
        max_intensity = np.max(heatmap)
        
        stats_text = f'Total Points: {total_points}\nClicks: {clicks}\nAvg Intensity: {avg_intensity:.3f}\nMax Intensity: {max_intensity:.3f}'
        plt.text(10, 30, stats_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 저장
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"[HEATMAP] 저장 완료: {output_path}")
        print(f"[HEATMAP] 통계 - 총 포인트: {total_points}, 클릭: {clicks}, 평균 강도: {avg_intensity:.3f}")
        
        return {
            'total_points': total_points,
            'clicks': clicks,
            'avg_intensity': float(avg_intensity),
            'max_intensity': float(max_intensity),
            'heatmap_shape': heatmap.shape
        }
        
    except Exception as e:
        print(f"[HEATMAP ERROR] {e}")
        return None

# ==========================
# Batch Processing Functions
# ==========================

def get_input_images(input_dir="omniparser/input"):
    """input 폴더에서 모든 이미지 파일을 찾아서 반환"""
    if not os.path.exists(input_dir):
        print(f"[WARNING] Input 디렉토리가 존재하지 않습니다: {input_dir}")
        return []
    
    # 지원하는 이미지 확장자
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(input_dir, ext)
        image_files.extend(glob.glob(pattern))
        # 대문자 확장자도 확인
        pattern_upper = os.path.join(input_dir, ext.upper())
        image_files.extend(glob.glob(pattern_upper))
    
    # 중복 제거 및 정렬
    image_files = sorted(list(set(image_files)))
    
    print(f"[BATCH] Input 폴더에서 {len(image_files)}개의 이미지 파일을 찾았습니다:")
    for i, img_path in enumerate(image_files, 1):
        print(f"  {i}. {os.path.basename(img_path)}")
    
    return image_files

def save_simulation_result(output_dir, image_name, result, target_texts=None):
    """시뮬레이션 결과를 output 폴더에 저장 (다중 task 지원)"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[BATCH] Output 디렉토리를 생성했습니다: {output_dir}")
    
    # 결과 데이터 구성 (다중 task 지원)
    result_data = {
        'timestamp': datetime.now().isoformat(),
        'image_name': image_name,
        'user_type': CONFIG['user_type'],
        'total_duration': result.get('total_duration', 0.0),
        'image_path': result.get('image_path', ''),
        
        # 평균 통계
        'summary': {
            'total_tasks': result.get('total_tasks', 0),
            'successful_tasks': result.get('successful_tasks', 0),
            'success_rate': result.get('success_rate', 0.0),
            'avg_steps': result.get('avg_steps', 0.0),
            'avg_clicks': result.get('avg_clicks', 0.0),
            'avg_duration': result.get('avg_duration', 0.0)
        },
        
        # 개별 task 결과
        'individual_tasks': result.get('individual_tasks', [])
    }
    
    # JSON 파일로 저장
    base_name = os.path.splitext(image_name)[0]
    output_file = os.path.join(output_dir, f"{base_name}_result.json")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"[BATCH] 결과 저장 완료: {output_file}")
        print(f"  - 총 {result_data['summary']['total_tasks']}개 task, 성공률: {result_data['summary']['success_rate']:.1f}%")
        return True
    except Exception as e:
        print(f"[ERROR] 결과 저장 실패: {e}")
        return False

def print_batch_summary(results):
    """배치 처리 결과 요약 출력 (다중 task 지원)"""
    if not results:
        print("[BATCH] 처리된 결과가 없습니다.")
        return
    
    total_images = len(results)
    valid_results = [r for r in results if r is not None]
    
    # 전체 통계 계산
    total_tasks = sum(r.get('summary', {}).get('total_tasks', 0) for r in valid_results)
    total_successful_tasks = sum(r.get('summary', {}).get('successful_tasks', 0) for r in valid_results)
    total_steps = sum(r.get('summary', {}).get('avg_steps', 0) * r.get('summary', {}).get('total_tasks', 0) for r in valid_results)
    total_clicks = sum(r.get('summary', {}).get('avg_clicks', 0) * r.get('summary', {}).get('total_tasks', 0) for r in valid_results)
    total_duration = sum(r.get('total_duration', 0.0) for r in valid_results)
    
    overall_success_rate = (total_successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    print("\n" + "="*80)
    print("📊 BATCH SIMULATION SUMMARY (MULTI-TASK)")
    print("="*80)
    print(f"🖼️  Total Images Processed: {total_images}")
    print(f"🎯 Total Tasks Executed: {total_tasks}")
    print(f"✅ Successful Tasks: {total_successful_tasks}")
    print(f"❌ Failed Tasks: {total_tasks - total_successful_tasks}")
    print(f"📈 Overall Success Rate: {overall_success_rate:.1f}%")
    print(f"🔄 Total Steps: {total_steps:.0f}")
    print(f"🖱️  Total Clicks: {total_clicks:.0f}")
    print(f"⏱️  Total Duration: {total_duration:.2f} seconds")
    print(f"⚡ Average Time per Image: {total_duration/total_images:.2f}s")
    if total_tasks > 0:
        print(f"🎯 Average Tasks per Image: {total_tasks/total_images:.1f}")
        print(f"🔄 Average Steps per Task: {total_steps/total_tasks:.1f}")
        print(f"🖱️  Average Clicks per Task: {total_clicks/total_tasks:.1f}")
    else:
        print(f"🎯 Average Tasks per Image: 0.0")
        print(f"🔄 Average Steps per Task: 0.0")
        print(f"🖱️  Average Clicks per Task: 0.0")
    
    # 개별 이미지 결과 상세
    print("\n📋 INDIVIDUAL IMAGE RESULTS:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        if result:
            summary = result.get('summary', {})
            success_rate = summary.get('success_rate', 0.0)
            total_tasks_img = summary.get('total_tasks', 0)
            successful_tasks_img = summary.get('successful_tasks', 0)
            avg_steps = summary.get('avg_steps', 0.0)
            avg_clicks = summary.get('avg_clicks', 0.0)
            duration = result.get('total_duration', 0.0)
            
            print(f"{i:2d}. {result.get('image_path', 'Unknown'):30s} | "
                  f"Tasks: {successful_tasks_img}/{total_tasks_img} ({success_rate:5.1f}%) | "
                  f"Avg Steps: {avg_steps:5.1f} | Avg Clicks: {avg_clicks:4.1f} | "
                  f"Time: {duration:6.2f}s")
        else:
            print(f"{i:2d}. {'ERROR':30s} | Tasks: N/A | Avg Steps: N/A | Avg Clicks: N/A | Time: N/A")
    
    print("="*80)

# ==========================
# Main
# ==========================

def run_single_simulation(image_path, target_texts=None, output_dir="omniparser/output"):
    """단일 이미지에서 다중 task 수행 (5개 task의 평균 결과 반환)"""
    try:
        # 전체 수행시간 측정 시작
        total_start_time = time.time()
        
        # 마우스 히스토리 수집용
        mouse_history = []
        
        def record_mouse_move(x, y, duration=0, event_type='move'):
            """마우스 이동을 기록하고 실행"""
            mouse_history.append((x, y, duration, event_type))
            return x, y
        
        def record_mouse_click(x, y, event_type='click'):
            """마우스 클릭을 기록하고 실행"""
            mouse_history.append((x, y, 0, event_type))
            return x, y
    
        # 사용자 행동 프로파일 로드
        user_behavior = get_user_behavior(CONFIG['user_type'])
        print(f"[BEHAVIOR] User type: {CONFIG['user_type']}")
        print(f"[BEHAVIOR] Curved movement: {user_behavior['enable_curved_movement']}")
        print(f"[BEHAVIOR] Complex click: {user_behavior['enable_complex_click']}")
        
        # 이미지 경로 설정
        if image_path is None:
            image_path = SCREEN_PATH
        
        # 파일 존재 확인 및 예외 처리
        if not os.path.exists(image_path):
            print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {image_path}")
            return None
        
        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
            return None
    
        if not os.path.exists(CLICK_SOUND_PATH):
            print(f"[WARNING] 클릭 사운드 파일을 찾을 수 없습니다: {CLICK_SOUND_PATH}")
            click_snd = None
        else:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=256)
                click_snd = pygame.mixer.Sound(CLICK_SOUND_PATH)
            except Exception as e:
                print(f"[WARNING] 사운드 초기화 실패: {e}")
                click_snd = None

        try:
            env = GazeKioskEnv(image_path, verbose=False) 
            env.max_steps = MAX_EP_STEPS
        except Exception as e:
            print(f"[ERROR] 환경 초기화 실패: {e}")
            _close_image_display()
            return None

        # 이미지 OCR 처리 함수 연결
        env._process_image_ocr = types.MethodType(_process_image_ocr, env)
        env.set_goal_sequence = types.MethodType(_set_goal_sequence, env)
        env._goal_step_postprocess = types.MethodType(_goal_step_postprocess, env)

        # 이미지 OCR 처리
        obs = env._process_image_ocr(image_path)
        obs_dim = obs.shape[0]
    
        try:
            net = GazeActorCritic(obs_dim).to(device)
            # 저장된 모델 로드 시도
            saved_state = torch.load(MODEL_PATH, map_location=device)
            
            # 차원 불일치 확인 및 처리
            if saved_state['net.0.weight'].shape[1] != obs_dim:
                print(f"[WARNING] 모델 차원 불일치: 저장된 모델 {saved_state['net.0.weight'].shape[1]} vs 현재 {obs_dim}")
                print("[INFO] 새로운 모델로 초기화합니다.")
                net = GazeActorCritic(obs_dim).to(device)  # 새로운 모델로 초기화
            else:
                net.load_state_dict(saved_state)
            
            net.eval()
            print(f"[SUCCESS] 모델 로드 완료 (차원: {obs_dim})")
        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {e}")
            print("[INFO] 새로운 모델로 초기화합니다.")
            net = GazeActorCritic(obs_dim).to(device)
            net.eval()

        # DPI 스케일 정보 출력
        screen_w, screen_h = pyautogui.size()
        print(f"[DPI INFO] pyautogui.size(): ({screen_w}, {screen_h})")
        print(f"[DPI INFO] SCREEN_MONITOR: {SCREEN_MONITOR}")
        if screen_w != SCREEN_MONITOR["width"] or screen_h != SCREEN_MONITOR["height"]:
            print(f"[DPI WARNING] 좌표계 불일치 감지! DPI 스케일 보정이 필요합니다.")
            print(f"  - pyautogui: {screen_w}x{screen_h}")
            print(f"  - mss: {SCREEN_MONITOR['width']}x{SCREEN_MONITOR['height']}")
        else:
            print(f"[DPI INFO] 좌표계 일치 - DPI 스케일 보정 불필요")

        # OCR 텍스트에서 랜덤 5개 task 선택
        if target_texts is None:
            selected_tasks = select_random_tasks(env.ocr_txt, env.ocr_bb, env.W, env.H, num_tasks=5)
            if not selected_tasks:
                print("[ERROR] 선택된 task가 없습니다.")
                return None
        else:
            # 외부에서 제공된 target_texts 사용
            selected_tasks = [{'text': text, 'bbox': None, 'priority': 0} for text in target_texts]
        
        print(f"\n🎯 SELECTED TASKS: {len(selected_tasks)}개")
        for i, task in enumerate(selected_tasks, 1):
            print(f"  {i}. '{task['text']}'")
        
        # 5개 task 순차 실행
        task_results = []
        
        for task_idx, task in enumerate(selected_tasks, 1):
            target_text = task['text']
            print(f"\n{'='*60}")
            print(f"🎯 TASK {task_idx}/{len(selected_tasks)}: '{target_text}'")
            print(f"{'='*60}")
            
            # 환경 초기화 (각 task마다)
            env.reset()
            obs = env._obs()
            
            # 단일 task 실행
            task_sequence = [target_text]
            env.set_goal_sequence(task_sequence)
            
            # 통계 변수
            total_clicks = 0
            step = 0
            task_start_time = time.time()
            
            while True:
                step += 1
                
                # 35초 타임아웃 체크
                current_time = time.time()
                elapsed_time = current_time - task_start_time
                if elapsed_time > TASK_TIMEOUT_SECONDS:
                    print(f"  ⏰ Task {task_idx} 시간 초과 ({TASK_TIMEOUT_SECONDS}초): '{target_text}' (Steps: {step}, Time: {elapsed_time:.1f}s)")
                    found = False  # 실패로 처리
                    break
                
                if DEBUG and step % 50 == 0:  # 50스텝마다 출력
                    print(f"  Step {step}... (Time: {elapsed_time:.1f}s)")

                # 현재 목표
                goal_tok = task_sequence[env.goal_idx] if env.goal_idx < len(task_sequence) else None
                
                # 디버그 창 업데이트
                create_debug_window(env, goal_tok, step, task_start_time, total_clicks)

                # 행동 선택
                act = biased_sample(net, obs, env, alpha=2.0)
                obs, _, done, info = env.step(act)

                # viewport 마우스 이동
                PADDING = 5
                vx1, vy1, vx2, vy2 = env._vbox()
                cx_raw = (vx1 + vx2) / 2
                cy_raw = (vy1 + vy2) / 2

                if (step % MOUSE_MOVE_INTERVAL == 0):
                    # DPI 스케일 보정 적용
                    x, y = to_screen_xy(cx_raw, cy_raw)
                    x = max(PADDING, min(x, pyautogui.size()[0] - 1 - PADDING))
                    y = max(PADDING, min(y, pyautogui.size()[1] - 1 - PADDING))
                    
                    # 사용자 타입에 따른 마우스 이동
                    if user_behavior['enable_curved_movement']:
                        move_mouse_curved(x, y, user_behavior, record_mouse_move)  # 고령자: 곡선 이동
                    else:
                        # 젊은이: 직선 이동 (속도 적용)
                        sx, sy = pyautogui.position()
                        dist = math.hypot(x - sx, y - sy)
                        # 최소 duration 0.05초, 최대 duration 0.5초 보장
                        duration = max(0.05, min(0.5, dist / user_behavior['mouse_speed_px_s']))
                        if DEBUG and step % 10 == 0:  # 10스텝마다 디버그 출력
                            print(f"[MOUSE SPEED] {CONFIG['user_type'].title()}: {user_behavior['mouse_speed_px_s']} px/s, dist: {dist:.1f}px, duration: {duration:.3f}s")
                        record_mouse_move(x, y, duration, 'move')
                        pyautogui.moveTo(x, y, duration=duration)

                # 목표 찾기
                in_view = [t for t, b in zip(env.ocr_txt, env.ocr_bb) if is_inside(b, env._vbox())]
                found = env._goal_step_postprocess(in_view)

                # 목표 찾으면 클릭
                if found:
                    total_clicks += 1
                    time.sleep(CLICK_DELAY)
                    
                    # 클릭 후보 선택
                    cand = _pick_click_candidate(env._last_candidates, normalize_token(goal_tok))
                    if cand:
                        _, bb = cand
                        cx_raw, cy_raw = _bbox_center(bb)
                    else:
                        # 후보가 없으면 시야 박스 중심
                        cx_raw, cy_raw = _bbox_center(env._vbox())

                    # DPI 스케일 보정 적용
                    click_x, click_y = to_screen_xy(cx_raw, cy_raw)
                    
                    # 목표 위치로 먼저 이동
                    if user_behavior['enable_curved_movement']:
                        move_mouse_curved(click_x, click_y, user_behavior, record_mouse_move)  # 고령자: 곡선 이동
                    else:
                        # 젊은이: 직선 이동 (속도 적용)
                        sx, sy = pyautogui.position()
                        dist = math.hypot(click_x - sx, click_y - sy)
                        # 최소 duration 0.05초, 최대 duration 0.5초 보장
                        duration = max(0.05, min(0.5, dist / user_behavior['mouse_speed_px_s']))
                        if DEBUG:
                            print(f"[CLICK SPEED] {CONFIG['user_type'].title()}: {user_behavior['mouse_speed_px_s']} px/s, dist: {dist:.1f}px, duration: {duration:.3f}s")
                        record_mouse_move(click_x, click_y, duration, 'move')
                        pyautogui.moveTo(click_x, click_y, duration=duration)
                    
                    # 잠시 대기 후 클릭
                    time.sleep(0.1)
                    
                    # 사용자 타입에 따른 클릭 행동
                    if user_behavior['enable_complex_click']:
                        # 고령자: 복잡한 클릭 (호버 + 팻핑거 + 더블클릭/롱프레스)
                        target_bb = bb if cand else None
                        elder_click(click_x, click_y, user_behavior, target_bb)
                    else:
                        # 젊은이: 단순 클릭
                        record_mouse_click(click_x, click_y, 'click')
                        pyautogui.click()
                    
                    if click_snd:
                        click_snd.play()
                    
                    print(f"  ✅ Task {task_idx} 완료: '{target_text}' (Steps: {step}, Clicks: {total_clicks})")
                    break

                if step >= MAX_EP_STEPS:
                    print(f"  ❌ Task {task_idx} 스텝 타임아웃: '{target_text}' (Steps: {step}, Time: {elapsed_time:.1f}s)")
                    found = False  # 실패로 처리
                    break
            
            # task 결과 저장
            task_end_time = time.time()
            task_duration = task_end_time - task_start_time
            
            task_result = {
                'success': found,
                'steps': step,
                'clicks': total_clicks,
                'duration': task_duration,
                'target': target_text,
                'task_index': task_idx
            }
            
            task_results.append(task_result)
            
            # task 간 잠시 대기
            if task_idx < len(selected_tasks):
                print(f"  [WAIT] 다음 task까지 1초 대기...")
                time.sleep(1.0)

        # 전체 수행시간 측정
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # 평균 결과 계산
        avg_result = calculate_task_averages(task_results)
        
        print(f"\n{'='*60}")
        print("📊 MULTI-TASK PERFORMANCE REPORT")
        print(f"{'='*60}")
        print(f"⏱️  Total Execution Time: {total_duration:.2f} seconds")
        print(f"🎯 Total Tasks: {len(selected_tasks)}")
        print(f"✅ Successful Tasks: {avg_result['successful_tasks']}")
        print(f"📈 Success Rate: {avg_result['success_rate']:.1f}%")
        print(f"🔄 Average Steps: {avg_result['avg_steps']:.1f}")
        print(f"🖱️  Average Clicks: {avg_result['avg_clicks']:.1f}")
        print(f"⚡ Average Time per Task: {avg_result['avg_duration']:.2f}s")
        
        # 개별 task 결과
        print(f"\n📋 INDIVIDUAL TASK RESULTS:")
        for i, result in enumerate(task_results, 1):
            if result['success']:
                status = "✅ SUCCESS"
            else:
                if result['duration'] > TASK_TIMEOUT_SECONDS:
                    status = "⏰ TIMEOUT"
                else:
                    status = "❌ FAILED"
            print(f"  {i}. '{result['target']:15s}' | {status:10s} | "
                  f"Steps: {result['steps']:3d} | Clicks: {result['clicks']:2d} | "
                  f"Time: {result['duration']:6.2f}s")
        
        print("="*60)
        print("✔ MULTI-TASK 완료")
        
        # 디버그 창 정리
        if SHOW_DEBUG_WINDOW and debug_window:
            debug_window.destroy()
        
        # 이미지 표시 창 닫기
        _close_image_display()
        
        # 메모리 정리
        import gc
        gc.collect()
        
        # Heatmap 생성
        heatmap_stats = None
        print(f"[HEATMAP DEBUG] mouse_history 개수: {len(mouse_history)}")
        if mouse_history:
            try:
                # 캡쳐된 화면 크기 사용 (마우스 좌표와 일치)
                captured_size = pyautogui.size()  # 실제 화면 크기
                print(f"[HEATMAP DEBUG] 화면 크기: {captured_size}")
                
                # Heatmap 파일 경로
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                # output 디렉토리가 존재하는지 확인하고 생성
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                heatmap_path = os.path.join(output_dir, f"{image_name}_heatmap.png")
                print(f"[HEATMAP DEBUG] 저장 경로: {heatmap_path}")
                
                # Heatmap 생성 (캡쳐된 화면 크기 기준)
                heatmap_stats = create_gaze_heatmap(
                    mouse_history, 
                    captured_size, 
                    heatmap_path, 
                    title=f"Gaze Heatmap - {image_name} (Captured Screen)"
                )
                
                if heatmap_stats:
                    print(f"[HEATMAP] 생성 완료: {heatmap_path}")
                else:
                    print(f"[HEATMAP] 생성 실패")
                    
            except Exception as e:
                print(f"[HEATMAP ERROR] {e}")
                import traceback
                print(f"[HEATMAP TRACEBACK] {traceback.format_exc()}")
        else:
            print(f"[HEATMAP DEBUG] mouse_history가 비어있어서 히트맵을 생성하지 않습니다.")
        
        # 평균 결과에 개별 결과도 포함해서 반환
        avg_result['individual_tasks'] = task_results
        avg_result['total_duration'] = total_duration
        avg_result['image_path'] = image_path
        avg_result['heatmap_stats'] = heatmap_stats
        avg_result['mouse_history_count'] = len(mouse_history)
        
        return avg_result
    
    except KeyboardInterrupt:
        print("\n[INTERRUPT] 사용자에 의해 중단됨")
        _close_image_display()
        return None
    except Exception as e:
        print(f"[ERROR] 예상치 못한 오류 발생: {e}")
        _close_image_display()
        return None

def main(input_dir="omniparser/input", output_dir="omniparser/output", target_texts=None):
    """배치 처리: input 폴더의 모든 이미지에 대해 다중 task 시뮬레이션 실행"""
    print("="*80)
    print("🚀 OMNIPARSER BATCH SIMULATION START (MULTI-TASK)")
    print("="*80)
    print(f"📁 Input Directory: {input_dir}")
    print(f"📁 Output Directory: {output_dir}")
    if target_texts:
        print(f"🎯 Target Texts: {target_texts}")
    else:
        print(f"🎯 Target Selection: Random 5 tasks from OCR (edge priority)")
    print(f"👤 User Type: {CONFIG['user_type']}")
    print("="*80)
    
    # input 폴더에서 이미지 파일들 찾기
    image_files = get_input_images(input_dir)
    
    if not image_files:
        print("[ERROR] 처리할 이미지 파일이 없습니다.")
        return
    
    # 배치 처리 시작
    batch_start_time = time.time()
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"\n{'='*60}")
        print(f"🖼️  Processing Image {i}/{len(image_files)}: {image_name}")
        print(f"{'='*60}")
        
        try:
            # subprocess로 별도 프로세스 실행 (Segmentation fault 격리)
            import subprocess
            import sys
            
            print(f"[PROCESS] {image_name}을 별도 프로세스로 실행...")
            
            # 이미지 경로를 상대 경로로 변환 (omniparser/input -> input)
            relative_image_path = image_path
            if image_path.startswith("omniparser/"):
                relative_image_path = image_path[11:]  # "omniparser/" 제거
            
            # output 디렉토리를 상대 경로로 변환 (omniparser/output -> output)
            relative_output_dir = output_dir
            if output_dir.startswith("omniparser/"):
                relative_output_dir = output_dir[11:]  # "omniparser/" 제거
            
            # 현재 스크립트를 subprocess로 실행
            cmd = [
                sys.executable, 
                __file__,  # 현재 스크립트 파일
                relative_image_path,  # 상대 이미지 경로
                relative_output_dir,  # 상대 출력 디렉토리
                "--standalone"  # 독립 실행 모드
            ]
            
            # subprocess 실행 시 작업 디렉토리를 스크립트 파일의 디렉토리로 설정
            script_dir = os.path.dirname(os.path.abspath(__file__))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, encoding='utf-8', errors='ignore', cwd=script_dir)  # 5분 타임아웃
            
            if result.returncode == 0:
                print(f"[SUCCESS] {image_name} 처리 완료")
                # 결과 파일에서 결과 로드 (스크립트 디렉토리 기준)
                result_file = os.path.join(script_dir, relative_output_dir, f"{os.path.splitext(image_name)[0]}_result.json")
                print(f"[DEBUG] 결과 파일 찾는 경로: {result_file}")
                if os.path.exists(result_file):
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    results.append(result_data)
                    
                    # 진행 상황 출력
                    success_rate = result_data.get('summary', {}).get('success_rate', 0.0)
                    total_tasks = result_data.get('summary', {}).get('total_tasks', 0)
                    successful_tasks = result_data.get('summary', {}).get('successful_tasks', 0)
                    avg_steps = result_data.get('summary', {}).get('avg_steps', 0.0)
                    avg_clicks = result_data.get('summary', {}).get('avg_clicks', 0.0)
                    total_duration = result_data.get('total_duration', 0.0)
                    
                    print(f"[PROGRESS] {i+1}/{len(image_files)} 완료 - {successful_tasks}/{total_tasks} tasks 성공 ({success_rate:.1f}%)")
                    print(f"  Avg Steps: {avg_steps:.1f}, Avg Clicks: {avg_clicks:.1f}, Time: {total_duration:.2f}s")
                else:
                    print(f"[WARNING] 결과 파일을 찾을 수 없습니다: {result_file}")
                    print(f"[DEBUG] 현재 작업 디렉토리: {os.getcwd()}")
                    print(f"[DEBUG] output_dir: {output_dir}")
                    print(f"[DEBUG] relative_output_dir: {relative_output_dir}")
                    # 절대 경로로 다시 시도
                    abs_result_file = os.path.abspath(result_file)
                    print(f"[DEBUG] 절대 경로로 시도: {abs_result_file}")
                    if os.path.exists(abs_result_file):
                        print(f"[DEBUG] 절대 경로에서 파일 발견!")
                        with open(abs_result_file, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)
                        results.append(result_data)
                    else:
                        results.append(None)
            else:
                print(f"[ERROR] {image_name} 처리 실패 (exit code: {result.returncode})")
                if result.stderr:
                    print(f"[PROCESS ERROR] {result.stderr}")
                if result.stdout:
                    print(f"[PROCESS OUTPUT] {result.stdout}")
                results.append(None)
                
        except subprocess.TimeoutExpired:
            print(f"[ERROR] {image_name} 처리 시간 초과 (5분)")
            results.append(None)
                
        except Exception as e:
            print(f"[ERROR] {image_name} 처리 중 예외 발생: {e}")
            results.append(None)
        
        # 다음 이미지 처리 전 잠시 대기
        if i < len(image_files):
            print(f"[WAIT] 다음 이미지 처리 전 2초 대기...")
            time.sleep(2.0)
    
    # 배치 처리 완료
    batch_end_time = time.time()
    batch_duration = batch_end_time - batch_start_time
    
    print(f"\n{'='*80}")
    print("🏁 BATCH SIMULATION COMPLETED")
    print(f"{'='*80}")
    print(f"⏱️  Total Batch Time: {batch_duration:.2f} seconds")
    
    # 결과 요약 출력
    print_batch_summary(results)
    
    # 전체 결과를 하나의 JSON 파일로도 저장
    try:
        summary_file = os.path.join(output_dir, "batch_summary.json")
        summary_data = {
            'batch_timestamp': datetime.now().isoformat(),
            'input_directory': input_dir,
            'output_directory': output_dir,
            'target_text': target_texts[0] if target_texts else 'Unknown',
            'user_type': CONFIG['user_type'],
            'total_images': len(image_files),
            'batch_duration': batch_duration,
            'results': results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"\n📄 전체 요약이 저장되었습니다: {summary_file}")
        
    except Exception as e:
        print(f"[WARNING] 요약 파일 저장 실패: {e}")
    
    return results

# ==========================
# Elder/Young Behavior Profiles
# ==========================

@dataclass
class ElderProfile:
    # 시야/스캔 (지각)
    fixation_mean_s: float = 0.35  # 0.55 -> 0.35로 감소
    fixation_sigma: float = 0.20   # 0.35 -> 0.20으로 감소
    saccade_latency_mean_s: float = 0.20  # 0.32 -> 0.20으로 감소
    saccade_latency_sigma: float = 0.15   # 0.30 -> 0.15로 감소
    micro_saccade_px: float = 1.0   # 2.0 -> 1.0으로 감소
    
    # 마우스 움직임 (운동)
    mouse_speed_px_s: float = 800   # 고령자: 느린 속도 (800 px/s)
    path_curvature: float = 0.08   # 0.18 -> 0.08로 감소
    tremor_std_px: float = 2.0     # 1.2 -> 2.0으로 증가 (떨림 증가)
    overshoot_prob: float = 0.15   # 0.35 -> 0.15로 감소
    overshoot_ratio: float = 0.03  # 0.06 -> 0.03으로 감소
    
    # 클릭 행동
    hover_mean_s: float = 0.25  # 0.65 -> 0.25로 감소
    hover_sigma: float = 0.15   # 0.35 -> 0.15로 감소
    fatfinger_std_px: float = 3.0  # 6.0 -> 3.0으로 감소
    double_click_prob: float = 0.05  # 0.12 -> 0.05로 감소
    long_press_prob: float = 0.03   # 0.10 -> 0.03으로 감소
    long_press_ms: tuple = (200, 400)  # (300,700) -> (200,400)으로 감소

@dataclass
class YoungProfile:
    # 시야/스캔 (지각)
    fixation_mean_s: float = 0.25
    fixation_sigma: float = 0.15
    saccade_latency_mean_s: float = 0.15
    saccade_latency_sigma: float = 0.10
    micro_saccade_px: float = 0.5
    
    # 마우스 움직임 (운동)
    mouse_speed_px_s: float = 3000  # 젊은이: 빠른 속도 (3000 px/s)
    path_curvature: float = 0.05
    tremor_std_px: float = 0.5
    overshoot_prob: float = 0.05
    overshoot_ratio: float = 0.02
    
    # 클릭 행동
    hover_mean_s: float = 0.15
    hover_sigma: float = 0.10
    fatfinger_std_px: float = 1.0
    double_click_prob: float = 0.02
    long_press_prob: float = 0.02
    long_press_ms: tuple = (100, 200)

# 전역 프로파일 인스턴스
ELDER_PROFILE = ElderProfile()
YOUNG_PROFILE = YoungProfile()

def _lognorm(mean, sigma):
    """로그정규분포 샘플링"""
    return float(np.random.lognormal(mean=np.log(max(mean, 1e-3)), sigma=sigma))

def get_user_behavior(user_type: str):
    """사용자 타입에 따른 행동 모델 반환"""
    if user_type == 'elder':
        profile = ELDER_PROFILE
        return {
            'fixation_mean_s': profile.fixation_mean_s,
            'fixation_sigma': profile.fixation_sigma,
            'mouse_speed_px_s': profile.mouse_speed_px_s,
            'path_curvature': profile.path_curvature,
            'tremor_std_px': profile.tremor_std_px,
            'overshoot_prob': profile.overshoot_prob,
            'overshoot_ratio': profile.overshoot_ratio,
            'hover_mean_s': profile.hover_mean_s,
            'hover_sigma': profile.hover_sigma,
            'fatfinger_std_px': profile.fatfinger_std_px,
            'double_click_prob': profile.double_click_prob,
            'long_press_prob': profile.long_press_prob,
            'long_press_ms': profile.long_press_ms,
            'enable_dwell_manager': True,
            'enable_micro_saccade': True,
            'enable_curved_movement': True,
            'enable_complex_click': True
        }
    else:  # young
        profile = YOUNG_PROFILE
        return {
            'fixation_mean_s': profile.fixation_mean_s,
            'fixation_sigma': profile.fixation_sigma,
            'mouse_speed_px_s': profile.mouse_speed_px_s,
            'path_curvature': profile.path_curvature,
            'tremor_std_px': profile.tremor_std_px,
            'overshoot_prob': profile.overshoot_prob,
            'overshoot_ratio': profile.overshoot_ratio,
            'hover_mean_s': profile.hover_mean_s,
            'hover_sigma': profile.hover_sigma,
            'fatfinger_std_px': profile.fatfinger_std_px,
            'double_click_prob': profile.double_click_prob,
            'long_press_prob': profile.long_press_prob,
            'long_press_ms': profile.long_press_ms,
            'enable_dwell_manager': False,
            'enable_micro_saccade': False,
            'enable_curved_movement': False,
            'enable_complex_click': False
        }

# ==========================
# Mouse Movement Functions
# ==========================
def move_mouse_curved(x, y, behavior, record_func=None):
    """곡선 경로로 마우스 이동 (고령자용)"""
    sx, sy = pyautogui.position()
    dist = math.hypot(x - sx, y - sy)
    duration = dist / max(80, behavior['mouse_speed_px_s'])
    
    if DEBUG:
        print(f"[MOUSE SPEED] Elder (Curved): {behavior['mouse_speed_px_s']} px/s, dist: {dist:.1f}px, duration: {duration:.3f}s")

    # 중간 제어점(베지어) - 경로를 살짝 휘게
    midx = (sx + x) / 2
    midy = (sy + y) / 2
    nx, ny = x - sx, y - sy
    
    if dist > 0:
        px, py = -ny/dist, nx/dist  # 수직 방향
        ctrlx = midx + px * behavior['path_curvature'] * dist
        ctrly = midy + py * behavior['path_curvature'] * dist
    else:
        ctrlx, ctrly = midx, midy

    steps = max(12, int(duration * 60))
    for i in range(1, steps + 1):
        t = i / steps
        bx = (1-t)**2 * sx + 2*(1-t)*t*ctrlx + t**2 * x
        by = (1-t)**2 * sy + 2*(1-t)*t*ctrly + t**2 * y
        bx += np.random.normal(0, behavior['tremor_std_px'])
        by += np.random.normal(0, behavior['tremor_std_px'])
        pyautogui.moveTo(int(bx), int(by))
        time.sleep(duration/steps)

    # 오버슈트 연출
    if random.random() < behavior['overshoot_prob'] and dist > 80:
        ox = x + int((x - sx) * behavior['overshoot_ratio'])
        oy = y + int((y - sy) * behavior['overshoot_ratio'])
        pyautogui.moveTo(ox, oy, duration=0.08)
        pyautogui.moveTo(x, y, duration=0.10)
    
    # 마우스 이동 기록
    if record_func:
        record_func(x, y, duration, 'move')

# ==========================
# Click Functions
# ==========================
def fitts_delay(distance_px, target_w_px):
    """Fitts's Law 기반 클릭 지연 계산"""
    a, b = 0.10, 0.12
    W = max(8.0, target_w_px)
    return a + b * math.log2(distance_px / W + 1.0)

def elder_click(x, y, behavior, target_bb=None):
    """고령자용 복잡한 클릭 행동 (이미 이동된 상태에서 클릭만)"""
    # 호버(주저)
    time.sleep(_lognorm(behavior['hover_mean_s'], behavior['hover_sigma']))

    # 팻핑거 (이미 이동된 위치에서 미세 조정)
    x += int(np.random.normal(0, behavior['fatfinger_std_px']))
    y += int(np.random.normal(0, behavior['fatfinger_std_px']))
    
    # 팻핑거로 인한 위치 조정
    pyautogui.moveTo(x, y, duration=0.05)

    # 목표 크기/거리 기반 추가 지연
    sx, sy = pyautogui.position()
    d = math.hypot(x - sx, y - sy)
    tw = 60
    if target_bb is not None:
        w, h, _ = _bbox_wh(target_bb)
        tw = max(w, h)
    time.sleep(fitts_delay(d, tw))

    # 더블클릭/롱프레스
    r = random.random()
    if r < behavior['double_click_prob']:
        pyautogui.click(clicks=2, interval=_lognorm(0.28, 0.25))
    else:
        pyautogui.mouseDown()
        if random.random() < behavior['long_press_prob']:
            time.sleep(random.uniform(*[v/1000 for v in behavior['long_press_ms']]))
        time.sleep(_lognorm(0.08, 0.20))
        pyautogui.mouseUp()

# ==========================
# Simplified Processing (Dwell Manager removed)
# ==========================

if __name__ == "__main__":
    # 독립 실행 모드 확인
    if len(sys.argv) > 3 and sys.argv[3] == "--standalone":
        # 독립 실행 모드: 단일 이미지 처리
        image_path = sys.argv[1]
        output_dir = sys.argv[2]
        
        print(f"[STANDALONE] 독립 실행 모드: {os.path.basename(image_path)}")
        
        try:
            # 단일 이미지 시뮬레이션 실행
            result = run_single_simulation(image_path, target_texts=None, output_dir=output_dir)
            
            if result:
                # 결과 저장
                save_simulation_result(output_dir, os.path.basename(image_path), result)
                print(f"[STANDALONE] 완료 - 성공률: {result.get('success_rate', 0.0):.1f}%")
                sys.exit(0)  # 성공
            else:
                print(f"[STANDALONE] 실패")
                sys.exit(1)  # 실패
                
        except Exception as e:
            import traceback
            print(f"[STANDALONE ERROR] {e}")
            print(f"[STANDALONE TRACEBACK] {traceback.format_exc()}")
            sys.exit(1)  # 실패
    
    else:
        # 배치 처리 실행
        # 사용법: python gaze_ppo_test_v9_3.py [input_dir] [output_dir] [target_text1,target_text2,...]
        
        # 명령행 인수 처리
        input_dir = sys.argv[1] if len(sys.argv) > 1 else "omniparser/input"
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "omniparser/output"
        target_texts_str = sys.argv[3] if len(sys.argv) > 3 else None
        
        # target_texts 파싱
        target_texts = None
        if target_texts_str:
            target_texts = [text.strip() for text in target_texts_str.split(',')]
            print(f"[INFO] 사용자 지정 target texts: {target_texts}")
        else:
            print(f"[INFO] OCR 기반 랜덤 5개 task 선택 모드")
        
        print(f"[INFO] 배치 처리 시작")
        print(f"[INFO] Input Directory: {input_dir}")
        print(f"[INFO] Output Directory: {output_dir}")
        
        # 배치 시뮬레이션 실행
        results = main(input_dir, output_dir, target_texts)
        
        if results:
            print(f"\n🎉 배치 처리가 완료되었습니다! {len(results)}개의 이미지를 처리했습니다.")
        else:
            print(f"\n❌ 배치 처리 중 오류가 발생했습니다.")
