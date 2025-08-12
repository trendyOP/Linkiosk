import cv2
import time
import numpy as np
import pygame
import pyautogui
import mss
import types
import difflib
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
from PIL import Image
from text_normalizer import normalize_token
from typing import List

# Windows DPI 스케일 문제 해결
try:
    import ctypes
    ctypes.windll.user32.SetProcessDPIAware()
    print("[DPI] SetProcessDPIAware enabled - pyautogui가 물리 픽셀 좌표 사용")
except Exception as e:
    print("[DPI] Failed to set DPI aware:", e)

import random
pyautogui.FAILSAFE = True  # Failsafe 비활성화

SEED = 42
random.seed(SEED); np.random.seed(SEED)

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
    VISION_GRID_N = 32
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
    """mss 좌표를 pyautogui 좌표로 변환 (DPI 스케일 보정)"""
    # 한번 계산
    screen_w, screen_h = pyautogui.size()
    mon_w, mon_h = SCREEN_MONITOR["width"], SCREEN_MONITOR["height"]
    sx = screen_w / mon_w
    sy = screen_h / mon_h
    
    # raw = mss/원본 이미지 좌표 (물리 픽셀)
    return (
        int(SCREEN_MONITOR["left"] * sx + x_raw * sx),
        int(SCREEN_MONITOR["top"]  * sy + y_raw * sy),
    )

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
        parser.read('omniparser/config.ini', encoding='utf-8')
        
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

# 파일 경로 (환경변수나 설정 파일로 이동 권장)
SCREEN_PATH = "screen5.png"
MODEL_PATH  = "omniparser/gaze_ppo_v9.pt"
CLICK_SOUND_PATH = "click.wav"

# 테스트 설정 - config에서 읽어온 queue 사용
if CONFIG['menu_queue']:
    TEST_TASKS = [CONFIG['menu_queue']]
else:
    TEST_TASKS = [["매장식사","아메리카노","주문담기","더담기","스무디","수박 주스", "주문담기", "더담기","베이커리","햄&치즈 샌드위치","주문담기", "결제하기", "확인","신용카드","대기","예"]]
MAX_EP_STEPS = 2000
ALLOW_EXTRA = 2

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
HASH_ROI = (0.10, 0.90, 0.10, 0.90)  # ROI 비율을 더 넓게 조정하여 화면 변화 감지 개선

# 동작 설정
MOUSE_MOVE_INTERVAL = 5  # 마우스 이동 간격
CLICK_DELAY = 1.143  # 클릭 후 대기 시간, 평균 노인의 인지시간 + 반응시간
UI_TRANSITION_DELAY = 0.05  # UI 전환 대기 시간
MEMORY_PRUNING_INTERVAL = 5  # 메모리 정리 간격

# 메모리 설정
MAX_MEMORY_SIZE = 2000  # 메모리 최대 크기 제한 (증가)
MEMORY_CAP_PER_TOKEN = 200  # 토큰당 최대 메모리 항목 수
MEMORY_TTL = 5000  # 메모리 항목 생명주기 (step) - 증가

# ==========================
# Stable aHash (ROI + Blur)
# ==========================
def stable_ahash(img, size=16, roi=HASH_ROI):
    """중앙 콘텐츠 ROI만 자르고 blur+downsample 해서 ahash 생성"""
    H, W = img.shape[:2]
    ry1, ry2, rx1, rx2 = roi
    y1, y2 = int(H * ry1), int(H * ry2)
    x1, x2 = int(W * rx1), int(W * rx2)

    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 더 민감한 해시를 위해 블러 최소화
    gray = cv2.GaussianBlur(gray, (3,3), 0.05)  # sigma를 0.1에서 0.05로 더 감소
    
    # 더 큰 크기로 다운샘플링하여 더 많은 정보 보존
    small = cv2.resize(gray, (size*2, size*2), interpolation=cv2.INTER_AREA)  # 16x16에서 32x32로 증가
    avg = small.mean()
    
    # 더 세밀한 해시를 위해 임계값 조정
    bits = (small > avg).astype(np.uint8).reshape(-1)
    
    if DEBUG:
        # 해시 생성 과정 디버깅
        print(f"[HASH DEBUG] ROI: ({x1},{y1}) to ({x2},{y2})")
        print(f"[HASH DEBUG] Crop size: {crop.shape}, Small size: {small.shape}")
        print(f"[HASH DEBUG] Average: {avg:.2f}")
        print(f"[HASH DEBUG] Hash length: {len(bits)}")
    
    return ''.join(str(int(b)) for b in bits)

# ==========================
# OCR Cache
# ==========================
class OcrCache:
    def __init__(self, max_size=1, hamming_thr=0):  # 캐시 거의 비활성화: 완전 일치만 허용
        self.store = OrderedDict()
        self.hamming_thr = hamming_thr
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    @staticmethod
    def hamming(a, b):
        if a is None or b is None or len(a) != len(b):
            return 1e9
        return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

    def get(self, h):
        best_key, best_dist = None, 1e9
        for k in self.store.keys():
            d = self.hamming(h, k)
            if d < best_dist:
                best_key, best_dist = k, d

        if best_dist <= self.hamming_thr:
            v = self.store.pop(best_key)
            self.store[best_key] = v
            self.hits += 1
            if DEBUG:
                print(f"[CACHE HIT] dist={best_dist}, hits={self.hits}, misses={self.misses}")
                print(f"  Current hash: {h}")
                print(f"  Best match:   {best_key}")
            return v

        self.misses += 1
        if DEBUG:
            print(f"[CACHE MISS] closest_dist={best_dist}, hits={self.hits}, misses={self.misses}")
            print(f"  Current hash: {h}")
            if best_key:
                print(f"  Closest:      {best_key}")
                print(f"  Diff bits:    {sum(1 for i, (a, b) in enumerate(zip(h, best_key)) if a != b)}")
        return None

    def put(self, h, value):
        if h in self.store:
            self.store.pop(h)
        self.store[h] = value
        while len(self.store) > self.max_size:
            self.store.popitem(last=False)  # LRU 제거

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
        
        # 기본 힌트 맵
        base_hint = env.hint_map[gy,gx]
        
        # 장기간 실행 시 탐색 다양성 증가
        exploration_bonus = 0.0
        if env.steps > 1000:
            # 방문 빈도가 낮은 영역에 보너스
            visit_count = env.visited[gy,gx]
            if visit_count == 0:
                exploration_bonus = 0.5  # 미방문 영역 보너스
            elif visit_count < 3:
                exploration_bonus = 0.2  # 적게 방문한 영역 보너스
        
        # 메모리 기반 탐색 강화
        memory_bonus = 0.0
        if hasattr(env, 'memory_manager') and env.memory_manager:
            # 현재 목표와 관련된 메모리가 있는 방향에 보너스
            if env.goal_idx < len(env._goal_seq):
                current_goal = env._goal_seq[env.goal_idx]
                mem_target = env.memory_manager.best(current_goal)
                if mem_target:
                    # 메모리 위치로의 방향 계산
                    mem_dx = mem_target.gx - env.gx
                    mem_dy = mem_target.gy - env.gy
                    # 현재 이동 방향이 메모리 방향과 일치하면 보너스
                    if (dx == np.sign(mem_dx) and dy == np.sign(mem_dy)):
                        memory_bonus = 0.3
        
        total_hint = base_hint + exploration_bonus + memory_bonus
        hint_bias.append(total_hint)
    
    hint_bias = torch.tensor(hint_bias, dtype=torch.float32, device=device)
    biased_logits = logits + alpha * hint_bias
    return torch.distributions.Categorical(logits=biased_logits).sample().item()

# ─── Memory Manager ───────────────────────────────────────
@dataclass
class Mem:
    gx: int; gy: int; step: int; quality: float

class MemoryManager:
    def __init__(self, cap=MEMORY_CAP_PER_TOKEN, ttl=MEMORY_TTL, max_size=MAX_MEMORY_SIZE):
        self.store = defaultdict(list)
        self.cap, self.ttl = cap, ttl
        self.max_size = max_size
        self.total_items = 0

    def _qual(self, dist_center: float, step: int) -> float:
        # step을 0~1 범위로 정규화 (5000스텝 기준)
        normalized_step = min(step / 5000.0, 1.0)
        return (1.0 - dist_center) * 0.7 + normalized_step * 0.3

    def update(self, tokens, gx, gy, step, dist_center):
        q = self._qual(dist_center, step)

        for t_raw in tokens:
            t = normalize_token(t_raw)
            if not t:
                continue

            lst = self.store[t]
            idx = next((i for i, m in enumerate(lst)
                        if m.gx == gx and m.gy == gy), None)

            if idx is not None:
                lst[idx].step    = step
                lst[idx].quality = max(lst[idx].quality, q)
            else:
                # 메모리 크기 제한 확인
                if self.total_items >= self.max_size:
                    self._prune_oldest()
                
                lst.append(Mem(gx, gy, step, q))
                self.total_items += 1

            lst.sort(key=lambda m: (-m.quality, -m.step))
            del lst[self.cap:]   

    def _prune_oldest(self):
        """가장 오래된 메모리 항목 제거"""
        oldest_step = float('inf')
        oldest_key = None
        
        for key, lst in self.store.items():
            if lst:
                step = min(m.step for m in lst)
                if step < oldest_step:
                    oldest_step = step
                    oldest_key = key
        
        if oldest_key and self.store[oldest_key]:
            # 가장 오래된 항목 제거
            self.store[oldest_key].pop(0)
            self.total_items -= 1
            if not self.store[oldest_key]:
                del self.store[oldest_key]

    def prune(self, cur_step):
        """메모리 정리 - TTL 기반 + 품질 기반 정리"""
        for tok in list(self.store.keys()):
            original_len = len(self.store[tok])
            
            # TTL 기반 정리
            self.store[tok] = [m for m in self.store[tok] if cur_step - m.step <= self.ttl]
            
            # 품질 기반 정리 (낮은 품질의 오래된 항목 제거)
            if len(self.store[tok]) > self.cap:
                # 품질과 최신성 기준으로 정렬
                self.store[tok].sort(key=lambda m: (m.quality, m.step), reverse=True)
                # 상위 cap개만 유지
                self.store[tok] = self.store[tok][:self.cap]
            
            removed_count = original_len - len(self.store[tok])
            self.total_items -= removed_count
            if not self.store[tok]: 
                del self.store[tok]

    def best(self, token: str, min_ratio: float = 0.90, allow_extra=2):
        tok_n = normalize_token(token)
        
        if DEBUG:
            print(f"[MEMORY SEARCH] '{token}' (normalized: '{tok_n}') 검색 중...")
            print(f"  - 저장된 토큰들: {list(self.store.keys())[:10]}")  # 상위 10개만 출력
        
        # 1. 정확 매칭
        for k, lst in self.store.items():
            if tok_n == normalize_token(k):
                if DEBUG:
                    print(f"  - 정확 매칭: '{k}' -> ({lst[0].gx}, {lst[0].gy})")
                return lst[0]
        
        # 2. 분리된 텍스트 매칭
        split_patterns = {
            '더담기': ['더', '담기'],
            '더 담기': ['더', '담기'],
            '햄치즈샌드위치': ['햄치즈', '샌드위치'],
            '햄&치즈샌드위치': ['햄&치즈', '샌드위치'],
        }
        
        for pattern, parts in split_patterns.items():
            if tok_n == pattern:
                # 모든 부분이 메모리에 있는지 확인
                found_parts = []
                for part in parts:
                    for k, lst in self.store.items():
                        if normalize_token(k) == part:
                            found_parts.append((k, lst[0]))
                            break
                
                if len(found_parts) == len(parts):
                    if DEBUG:
                        print(f"  - 분리된 텍스트 매칭: '{pattern}' -> {[(k, f'({m.gx},{m.gy})') for k, m in found_parts]}")
                    # 첫 번째 부분의 위치 반환
                    return found_parts[0][1]
        
        # 3. 단어 경계/가격 매칭
        for k, lst in self.store.items():
            kn = normalize_token(k)
            if _is_word_boundary_match(tok_n, kn) or _prefix_with_price(tok_n, kn):
                if DEBUG:
                    print(f"  - 경계/가격 매칭: '{k}' -> ({lst[0].gx}, {lst[0].gy})")
                return lst[0]
        
        # 3. fuzzy 매칭
        best_key, best_score = None, 0.0
        for k in self.store.keys():
            kn = normalize_token(k)
            if tok_n in kn and len(kn) - len(tok_n) <= allow_extra:
                score = difflib.SequenceMatcher(None, tok_n, kn).ratio()
                if score > best_score:
                    best_key, best_score = k, score
        
        if best_key and best_score >= min_ratio:
            if DEBUG:
                print(f"  - fuzzy 매칭: '{best_key}' (score: {best_score:.2f}) -> ({self.store[best_key][0].gx}, {self.store[best_key][0].gy})")
            return self.store[best_key][0]
        
        if DEBUG:
            print(f"  - 매칭 실패")
        return None

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
# Cached refresh (NO force)
# ==========================
def _refresh_screen_cached(self: GazeKioskEnv, bgr_img: np.ndarray, cache: OcrCache):
    t0 = time.perf_counter()
    
    # 캐시를 완전히 비활성화하고 항상 OCR 실행
    if DEBUG:
        print(f"[OCR FORCE] Cache disabled - always performing OCR scan...")
    
    self.image = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    self.W, self.H = self.image.size
    self.cell_w, self.cell_h = self.W / self.N, self.H / self.N

    self.ocr_txt, self.ocr_bb = check_ocr_box(
        self.image,
        display_img=False,
        output_bb_format="xyxy",
        use_paddleocr=True,
        ocr_engine='paddleocr',
        roi=(0.0, 1.0, 0.0, 1.0),       # 전체 화면으로 변경 (캐시 효율성)
        paddle_det_limit_side_len=1920,  # 1080p~1440p 화면에 최적
        paddle_rec_score_thresh=0.35,    # 임계값을 낮춤 (한국어 텍스트 구제)
        paddle_lang='korean'             # 한국어로 변경 (매장식사 인식 향상)
    )

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
        print(f"[OCR RUN]   {time.perf_counter()-t0:.3f}s")
    return self._obs()

# ==========================
# Debug Window Functions (Tkinter 기반)
# ==========================
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

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
        
        self.progress_label = tk.Label(self.frame, text="Progress: 0/0", **label_style)
        self.progress_label.grid(row=2, column=0, sticky=tk.W, pady=3)
        
        self.gaze_label = tk.Label(self.frame, text="Gaze: (0, 0)", **label_style)
        self.gaze_label.grid(row=3, column=0, sticky=tk.W, pady=3)
        
        self.memory_label = tk.Label(self.frame, text="Memory: No", **label_style)
        self.memory_label.grid(row=4, column=0, sticky=tk.W, pady=3)
        
        # 클릭 수 정보 라벨 추가
        self.clicks_label = tk.Label(self.frame, text="Clicks: 0", **label_style)
        self.clicks_label.grid(row=5, column=0, sticky=tk.W, pady=3)
        
        # 상태 표시 (다크 테마)
        self.status_label = tk.Label(self.frame, text="Status: Ready", 
                                   bg='#2b2b2b', fg='#00ff00', font=('Arial', 12, 'bold'))
        self.status_label.grid(row=6, column=0, sticky=tk.W, pady=12)
        
        # 로그 텍스트 영역 (다크 테마)
        self.log_text = tk.Text(self.frame, height=10, width=60, 
                               font=('Consolas', 9), bg='#1e1e1e', fg='#ffffff',
                               insertbackground='#ffffff', selectbackground='#404040')
        self.log_text.grid(row=7, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # 스크롤바 (다크 테마)
        scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.log_text.yview,
                               bg='#404040', troughcolor='#2b2b2b')
        scrollbar.grid(row=7, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # 종료 버튼 (다크 테마)
        self.close_button = tk.Button(self.frame, text="Close", command=self.root.destroy,
                                    bg='#404040', fg='#ffffff', font=('Arial', 10),
                                    relief='flat', padx=20, pady=5)
        self.close_button.grid(row=8, column=0, pady=8)
        
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        
    def update_info(self, step, goal_tok, completed_goals, total_goals, env, mem_target, start_time, total_clicks):
        if not SHOW_DEBUG_WINDOW or not hasattr(self, 'root'):
            return
            
        try:
            elapsed_time = time.time() - start_time
            self.time_label.config(text=f"Time: {elapsed_time:.1f}s")
            self.goal_label.config(text=f"Goal: {goal_tok or 'None'}")
            self.progress_label.config(text=f"Progress: {completed_goals}/{total_goals}")
            self.gaze_label.config(text=f"Gaze: ({env.gx}, {env.gy})")
            self.memory_label.config(text=f"Memory: {'Yes' if mem_target else 'No'}")
            
            # 클릭 수 정보 업데이트
            self.clicks_label.config(text=f"Clicks: {total_clicks}")
            
            # 상태 업데이트 (다크 테마)
            if mem_target:
                self.status_label.config(text="Status: Using Memory", fg="#00ff00")  # 밝은 초록
            else:
                self.status_label.config(text="Status: Exploring", fg="#ffaa00")  # 주황
                
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

def create_debug_window(img, env, goal_tok, mem_target, step, completed_goals, total_goals, start_time, total_clicks):
    """디버그 창 생성 및 업데이트"""
    global debug_window
    
    if not SHOW_DEBUG_WINDOW:
        return
    
    # 첫 실행시 창 생성
    if debug_window is None:
        debug_window = DebugWindow()
    
    # 정보 업데이트
    debug_window.update_info(step, goal_tok, completed_goals, total_goals, env, mem_target, start_time, total_clicks)

# ==========================
# Main
# ==========================
import os
import math

def main():
    # 전체 수행시간 측정 시작
    total_start_time = time.time()
    
    # 파일 존재 확인 및 예외 처리
    if not os.path.exists(SCREEN_PATH):
        print(f"[ERROR] 스크린샷 파일을 찾을 수 없습니다: {SCREEN_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return
    
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
        env = GazeKioskEnv(SCREEN_PATH, verbose=False) 
        env.max_steps = MAX_EP_STEPS
    except Exception as e:
        print(f"[ERROR] 환경 초기화 실패: {e}")
        return

    cache = OcrCache(max_size=32, hamming_thr=8)  # 해시가 16×16 비트에 ROI가 넓게 잡혀 있으므로, 작은 UI 변화로도 해시가 흔들림
    env.cache = cache  # env에 cache 속성 추가
    env._refresh_screen_cached = types.MethodType(_refresh_screen_cached, env)
    env.set_goal_sequence = types.MethodType(_set_goal_sequence, env)
    env._goal_step_postprocess = types.MethodType(_goal_step_postprocess, env)
    
    # 메모리 관리자를 환경에 연결
    memory = MemoryManager()
    env.memory_manager = memory

    obs_dim = env.reset().shape[0]
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

    mon = SCREEN_MONITOR
    sct = mss.mss()

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

    seek_mode = False
    PRUNE_EVERY = MEMORY_PRUNING_INTERVAL

    # 전체 통계 변수
    total_goals = 0
    completed_goals = 0
    total_steps = 0
    total_clicks = 0  # 클릭 수 추적

    for tid, seq in enumerate(TEST_TASKS, 1):
        print(f"\n== TASK {tid}: {seq}")
        total_goals += len(seq)
        
        task_start_time = time.time()
        env.set_goal_sequence(seq); obs = env.reset(); step = 0
        need_refresh = True

        while True:
            step += 1
            total_steps += 1
            print("step : ", step)

            # 메모리 타깃
            goal_tok = seq[env.goal_idx] if env.goal_idx < len(seq) else None
            
            # "대기" goal인 경우 프로그램 완전 정지
            if goal_tok == "대기":
                print(f"[WAIT] 프로그램 정지하고 5초 대기 실행")
                found = env._goal_step_postprocess([])  # 빈 리스트로 호출
                if found:
                    completed_goals += 1
                    need_refresh = True  # 대기 완료 후 다음 goal을 위해 OCR 새로 수행
                    continue
                else:
                    # 대기 중에는 아무것도 하지 않고 대기만
                    time.sleep(0.1)  # 짧은 대기로 CPU 사용량 줄임
                    continue
            
            # 필요할 때만 OCR
            if need_refresh:
                scr = np.array(sct.grab(mon))[:, :, :3]
                obs = env._refresh_screen_cached(scr, cache)
                need_refresh = False

                if DEBUG:
                    cv2.imwrite("debug_capture.png", scr)
                    print(f"[OCR RESULT] 총 {len(env.ocr_txt)}개 텍스트 발견:")
                    for i, txt in enumerate(env.ocr_txt):  # 모든 텍스트 출력
                        txt_norm = normalize_token(txt)
                        print(f"  {i+1:2d}. '{txt}' -> '{txt_norm}'")
                    
                    # OCR 결과 시각화 (디버그용)
                    visualize_ocr_boxes(scr, env.ocr_txt, env.ocr_bb, "ocr_vis.png")
                    
                    print(f"[MEMORY STATUS] 현재 메모리 상태:")
                    for token, mem_list in env.memory_manager.store.items():
                        print(f"  '{token}': {len(mem_list)}개 위치 - {[(m.gx,m.gy) for m in mem_list[:3]]}")
                        if len(mem_list) > 3:
                            print(f"    ... 외 {len(mem_list)-3}개 더")
                
            # mem_target 미리 정의 (디버그 창에서 사용하기 위해)
            mem_target = env.memory_manager.best(goal_tok) if goal_tok else None
            
            # 디버그 창 업데이트
            create_debug_window(scr, env, goal_tok, mem_target, step, completed_goals, total_goals, task_start_time, total_clicks)

            # [추가] 메모리 좌표가 현재 화면에서 유효한지 검증
            if mem_target:
                target_gx, target_gy = mem_target.gx, mem_target.gy
                
                if DEBUG:
                    print(f"[MEMORY CHECK] '{goal_tok}' -> mem_target: ({target_gx},{target_gy})")
                    print(f"  - token_cells[{target_gy},{target_gx}] = {env.token_cells[target_gy, target_gx]}")
                
                # 해당 좌표에 실제로 텍스트 셀이 있는지 확인
                if not env.token_cells[target_gy, target_gx]:
                    print(f"[MEMORY INVALID] No text at ({target_gx},{target_gy}) on current screen")
                    print(f"  - 화면이 변경되어 메모리 위치가 무효화됨")
                    mem_target = None
                    seek_mode = False
                else:
                    # 추가 검증: 해당 위치에 실제로 goal과 매칭되는 텍스트가 있는지 확인
                    vx1, vy1, vx2, vy2 = env._vbox()
                    cell_center_x = (target_gx + 0.5) * env.cell_w
                    cell_center_y = (target_gy + 0.5) * env.cell_h
                    
                    # 해당 셀에 있는 텍스트들 확인
                    cell_texts = []
                    for txt, bb in zip(env.ocr_txt, env.ocr_bb):
                        if is_inside(bb, (cell_center_x - env.cell_w/2, cell_center_y - env.cell_h/2, 
                                         cell_center_x + env.cell_w/2, cell_center_y + env.cell_h/2)):
                            cell_texts.append(txt)
                    
                    if DEBUG:
                        print(f"  - 셀 내 텍스트들: {cell_texts}")
                    
                    # goal과 매칭되는 텍스트가 있는지 확인
                    goal_norm = normalize_token(goal_tok)
                    matching_in_cell = any(
                        normalize_token(txt) == goal_norm or 
                        _is_word_boundary_match(goal_norm, normalize_token(txt)) or
                        _prefix_with_price(goal_norm, normalize_token(txt))
                        for txt in cell_texts
                    )
                    
                    if not matching_in_cell:
                        print(f"[MEMORY MISMATCH] 셀에 '{goal_tok}'와 매칭되는 텍스트 없음")
                        print(f"  - 셀 텍스트: {cell_texts}")
                        mem_target = None
                        seek_mode = False
                    else:
                        print(f"[MEMORY VALID] '{goal_tok}' 메모리 위치 유효함")
                        seek_mode = True
            else:
                if DEBUG:
                    print(f"[MEMORY MISS] '{goal_tok}'에 대한 메모리 없음")
                seek_mode = False

            if mem_target:
                act = plan_move((env.gx, env.gy), (mem_target.gx, mem_target.gy))
            else:
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
                pyautogui.moveTo(x, y, duration=0.1)

            # 시야 박스 좌표 디버그 출력
            vbox = env._vbox()
            if DEBUG and step % 5 == 0:  # 5스텝마다 출력
                print(f"[VIEWPORT] 시야 박스: {vbox}")
                print(f"[VIEWPORT] 현재 위치: ({env.gx}, {env.gy}) -> ({env.pos[0]:.3f}, {env.pos[1]:.3f})")
                # '아메리카노' 좌표 확인
                for i, (txt, bb) in enumerate(zip(env.ocr_txt, env.ocr_bb)):
                    if '아메리카노' in txt:
                        print(f"[VIEWPORT] '아메리카노' 좌표: {bb} -> 시야 박스 내: {is_inside(bb, vbox)}")
            
            # 메모리 업데이트
            in_view = [t for t, b in zip(env.ocr_txt, env.ocr_bb) if is_inside(b, env._vbox())]
            found = env._goal_step_postprocess(in_view)

            cx, cy = (env._vbox()[0] + env._vbox()[2]) // 2, (env._vbox()[1] + env._vbox()[3]) // 2
            dist_center = (((cx - env.W/2) / env.W) ** 2 + ((cy - env.H/2) / env.H) ** 2)
            
            if DEBUG and in_view:
                print(f"[MEMORY UPDATE] 위치({env.gx},{env.gy})에서 {len(in_view)}개 텍스트 발견:")
                for txt in in_view:
                    print(f"  - '{txt}'")
            
            env.memory_manager.update(in_view, env.gx, env.gy, step, dist_center)
            
            # 메모리 상태 디버그 출력
            if DEBUG and step % 10 == 0:  # 10스텝마다 출력
                print(f"[MEMORY STATUS] 현재 메모리 상태:")
                for token, memories in env.memory_manager.store.items():
                    if memories:  # 메모리가 있는 토큰만 출력
                        print(f"  '{token}': {[(m.gx, m.gy) for m in memories[:3]]}")  # 상위 3개만
                print(f"[MEMORY STATUS] 총 {len(env.memory_manager.store)}개 토큰, {env.memory_manager.total_items}개 메모리")
            
            # miss_click 체크: 현재 goal과 비슷한 텍스트가 시야에 있는지 확인
            if goal_tok and in_view:
                goal_norm = normalize_token(goal_tok)
                
                # 사용자 타입에 따른 miss_click 확률 결정
                if CONFIG['user_type'] == 'elder':
                    miss_click_prob = CONFIG['elder_miss_click_prob']
                else:
                    miss_click_prob = CONFIG['young_miss_click_prob']
                
                for txt in in_view:
                    txt_norm = normalize_token(txt)
                    # 비슷한 텍스트가 있지만 goal과는 다른 경우
                    if (txt_norm != goal_norm and 
                        (_is_word_boundary_match(goal_norm, txt_norm) or 
                         _prefix_with_price(goal_norm, txt_norm) or
                         (goal_norm in txt_norm and _better_match(goal_norm, txt_norm)))):
                        
                        # 확률적으로 miss_click 발생
                        if random.random() < miss_click_prob:
                            total_clicks += 1
                            if DEBUG:
                                print(f"[MISS CLICK] '{goal_tok}'와 비슷한 '{txt}' 발견 (확률: {miss_click_prob:.2f}) - 클릭 수 증가: {total_clicks}")
                            break  # 한 번에 하나의 miss_click만 증가
                        else:
                            if DEBUG:
                                print(f"[MISS CLICK AVOIDED] '{goal_tok}'와 비슷한 '{txt}' 발견했지만 miss_click 발생하지 않음 (확률: {miss_click_prob:.2f})")

            # 목표 찾으면 클릭 + (변경되었을 수 있으니) 다시 캐시 조회
            if found:
                completed_goals += 1
                total_clicks += 1  # goal 클릭 수 증가
                time.sleep(CLICK_DELAY)
                cand = _pick_click_candidate(env._last_candidates, normalize_token(goal_tok))
                if cand:
                    _, bb = cand
                    cx_raw, cy_raw = _bbox_center(bb)
                else:
                    cx_raw, cy_raw = _bbox_center(env._vbox())

                # DPI 스케일 보정 적용
                click_x, click_y = to_screen_xy(cx_raw, cy_raw)
                
                print(f"[CLICK DEBUG] 원본 좌표: ({cx_raw:.1f}, {cy_raw:.1f}) -> 변환 좌표: ({click_x}, {click_y})")
                pyautogui.moveTo(click_x, click_y, duration=0.15)
                pyautogui.click()
                if click_snd:
                    click_snd.play()
                print(f"Clicked at {(click_x, click_y)} for token '{goal_tok}' (Total clicks: {total_clicks})")

                time.sleep(UI_TRANSITION_DELAY)  # UI 전환 대기
                scr = np.array(sct.grab(SCREEN_MONITOR))[:, :, :3]
                obs = env._refresh_screen_cached(scr, cache)  # force 없이!
                need_refresh = False
                seek_mode = False
                
                # 클릭 후 디버그 창 업데이트 (10스텝마다만)
                if step % 10 == 0:
                    create_debug_window(scr, env, goal_tok, mem_target, step, completed_goals, total_goals, task_start_time, total_clicks)
                continue

            if step % MEMORY_PRUNING_INTERVAL == 0:
                env.memory_manager.prune(step)

            if step >= MAX_EP_STEPS or env.goal_idx == len(seq):
                task_end_time = time.time()
                task_duration = task_end_time - task_start_time
                msg = "ALL GOALS FOUND" if env.goal_idx == len(seq) else "Episode ended (timeout)"
                print("mem:", {k:[(m.gx,m.gy) for m in v] for k,v in env.memory_manager.store.items()})
                print(f"  >> {msg} (Task duration: {task_duration:.2f}s)")
                break

    # 전체 수행시간 측정 및 통계 출력
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print("\n" + "="*60)
    print("📊 FINAL PERFORMANCE REPORT")
    print("="*60)
    print(f"⏱️  Total Execution Time: {total_duration:.2f} seconds")
    print(f"🎯 Total Goals: {total_goals}")
    print(f"✅ Completed Goals: {completed_goals}")
    print(f"❌ Failed Goals: {total_goals - completed_goals}")
    print(f"📈 Success Rate: {(completed_goals/total_goals*100):.1f}%")
    print(f"🔄 Total Steps: {total_steps}")
    print(f"🖱️  Total Clicks: {total_clicks}")
    print(f"⚡ Average Time per Goal: {total_duration/total_goals:.2f}s")
    print(f"🚀 Average Steps per Goal: {total_steps/total_goals:.1f}")
    print(f"🎯 Average Clicks per Goal: {total_clicks/total_goals:.1f}")
    print("="*60)
    print("✔ 모든 TASK 종료")
    
    # 디버그 창 정리
    if SHOW_DEBUG_WINDOW and debug_window:
        debug_window.destroy()

if __name__ == "__main__":
    main()
