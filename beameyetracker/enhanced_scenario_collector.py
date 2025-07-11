#!/usr/bin/env python3
"""
향상된 키오스크 시나리오 데이터 수집기
스크린샷과 시선 데이터를 시간과 함께 동기화하여 정확한 히트맵 생성
"""

import os
import sys
import time
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket
import requests
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple
import cv2
from PIL import ImageGrab, Image
import io
import base64
import mss

# 시선 추적 관련
try:
    from eyeware import beam_eye_tracker as bet
    from screeninfo import get_monitors
    EYEWARE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Eyeware SDK not available. Eye tracking will be disabled. Error: {e}")
    EYEWARE_AVAILABLE = False

# 키보드 입력 감지
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError as e:
    print(f"Warning: keyboard module not available. Install with: pip install keyboard. Error: {e}")
    KEYBOARD_AVAILABLE = False

class EnhancedKioskScenarioCollector:
    def __init__(self, scenario_name: str = "kiosk_scenario"):
        """
        향상된 키오스크 시나리오 데이터 수집기 초기화
        스크린샷과 시선 데이터를 시간과 함께 동기화
        """
        self.scenario_name = scenario_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 데이터 저장 디렉토리
        self.data_dir = f"kiosk_data/{scenario_name}_{self.timestamp}"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 로깅 설정
        self.setup_logging()
        
        # 시선 추적 관련
        self.gaze_data = []
        self.step_gaze_data = {}  # 단계별 시선 데이터
        self.tracker = None
        self.tracking_active = False
        
        # 클릭 이벤트 관련
        self.click_events = []
        self.step_click_events = {}  # 단계별 클릭 이벤트
        self.current_scenario_step = 0
        
        # 스크린샷 캡처 관련
        self.screenshot_data = []  # 스크린샷 데이터 저장
        self.screenshot_interval = 1.0  # 1초마다 스크린샷 캡처
        self.screenshot_active = False
        self.screenshot_thread = None
        
        # 오류 로그
        self.error_logs = []
        
        # 실시간 시나리오 표시 관련
        self.scenario_display_active = False
        self.scenario_display_thread = None
        
        # 단계별 시각화 관련
        self.step_visualizations_created = set()  # 이미 생성된 단계 시각화 추적
        
        # 헬 난이도 시나리오 정의
        self.scenarios = {
            "R1_kiosk_hell": [
                {
                    "step": 0,
                    "description": "베스트·신메뉴 카테고리에서 '흑당 콜드브루' 선택",
                    "target_element": "흑당 콜드브루",
                    "expected_action": "click",
                    "category": "best",
                    "item_id": "best_03",
                    "difficulty": "easy"
                },
                {
                    "step": 1,
                    "description": "커피 카테고리로 이동하여 '카라멜 마끼아또' 선택",
                    "target_element": "카라멜 마끼아또",
                    "expected_action": "click",
                    "category": "coffee",
                    "item_id": "cof_cm",
                    "difficulty": "medium"
                },
                {
                    "step": 2,
                    "description": "논-커피 음료 카테고리에서 '민트 초코 라떼' 선택",
                    "target_element": "민트 초코 라떼",
                    "expected_action": "click",
                    "category": "noncoffee",
                    "item_id": "nc_mint",
                    "difficulty": "hard"
                }
            ]
        }
        
        # 현재 시나리오
        self.current_scenario = "R1_kiosk_hell"
        self.scenario_steps = self.scenarios[self.current_scenario]
        
        # 화면 해상도
        self.screen_width, self.screen_height = self.get_screen_resolution()
        
        # 웹서버 설정
        self.server_port = self.find_free_port()
        self.server_thread = None
        
        # 실시간 시나리오 표시 설정
        self.scenario_display_port = self.find_free_port()
        self.scenario_display_thread = None
        
        self.logger.info(f"향상된 키오스크 시나리오 수집기 초기화 완료: {scenario_name}")
    
    def setup_logging(self):
        """로깅 시스템 설정"""
        log_file = os.path.join(self.data_dir, "collection.log")
        
        # 로거 설정
        self.logger = logging.getLogger('EnhancedKioskCollector')
        self.logger.setLevel(logging.DEBUG)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"로깅 시스템 초기화: {log_file}")
    
    def get_screen_resolution(self) -> Tuple[int, int]:
        """화면 해상도 가져오기"""
        try:
            if EYEWARE_AVAILABLE:
                primary_monitor = next((m for m in get_monitors() if m.is_primary), get_monitors()[0])
                width, height = primary_monitor.width, primary_monitor.height
                self.logger.info(f"화면 해상도 감지: {width}x{height}")
                return width, height
            else:
                self.logger.warning("시선 추적이 비활성화되어 기본 해상도를 사용합니다.")
                return 1920, 1080
        except Exception as e:
            self.logger.error(f"화면 해상도 감지 실패: {e}")
            return 1920, 1080  # 기본값
    
    def find_free_port(self) -> int:
        """사용 가능한 포트 찾기"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
                self.logger.info(f"사용 가능한 포트 찾음: {port}")
                return port
        except Exception as e:
            self.logger.error(f"포트 찾기 실패: {e}")
            return 8080  # 기본 포트
    
    def start_screenshot_capture(self):
        """스크린샷 캡처 시작"""
        try:
            self.screenshot_active = True
            self.screenshot_thread = threading.Thread(target=self._screenshot_capture_loop)
            self.screenshot_thread.daemon = True
            self.screenshot_thread.start()
            self.logger.info("스크린샷 캡처 시작")
            print("📸 스크린샷 캡처 시작")
        except Exception as e:
            self.logger.error(f"스크린샷 캡처 시작 실패: {e}")
    
    def stop_screenshot_capture(self):
        """스크린샷 캡처 중지"""
        try:
            self.screenshot_active = False
            if self.screenshot_thread:
                self.screenshot_thread.join(timeout=1.0)
            self.logger.info("스크린샷 캡처 중지")
        except Exception as e:
            self.logger.error(f"스크린샷 캡처 중지 실패: {e}")
    
    def _screenshot_capture_loop(self):
        """스크린샷 캡처 루프"""
        while self.screenshot_active:
            try:
                # 현재 시간
                current_time = time.time()
                
                # 스크린샷 캡처
                screenshot = ImageGrab.grab()
                
                # 이미지를 바이트로 변환
                img_byte_arr = io.BytesIO()
                screenshot.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Base64 인코딩
                img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
                
                # 스크린샷 데이터 저장
                screenshot_data = {
                    'timestamp': current_time,
                    'datetime': datetime.now().isoformat(),
                    'step': self.current_scenario_step,
                    'image_base64': img_base64,
                    'screen_resolution': {'width': self.screen_width, 'height': self.screen_height}
                }
                
                self.screenshot_data.append(screenshot_data)
                
                # 로그 (너무 자주 출력하지 않도록)
                if len(self.screenshot_data) % 10 == 0:
                    self.logger.info(f"스크린샷 캡처: {len(self.screenshot_data)}개")
                
                time.sleep(self.screenshot_interval)
                
            except Exception as e:
                self.logger.error(f"스크린샷 캡처 실패: {e}")
                time.sleep(1.0)
    
    def capture_left_monitor(self):
        """mss로 왼쪽(가장 x값이 작은) 모니터만 캡처 + 모니터 정보/캡처 bbox 로그 출력"""
        from PIL import Image
        with mss.mss() as sct:
            monitors = sct.monitors  # [0]은 전체, [1]부터 각 모니터
            for m in monitors[1:]:
                self.logger.info(f"mss 모니터: {m}")
            left_monitor = sorted(monitors[1:], key=lambda m: m['left'])[0]
            self.logger.info(f"mss 왼쪽 모니터 bbox: {left_monitor}")
            img = sct.grab(left_monitor)
            img_pil = Image.frombytes('RGB', img.size, img.rgb)
            return img_pil

    def advance_scenario_step(self):
        """다음 단계로 진행 (구간별 시선/스크린샷/머문 시간 저장, 왼쪽 모니터만 캡처)"""
        if self.current_scenario_step < len(self.scenario_steps):
            # 1. 현재 시점 스크린샷 저장 (왼쪽 모니터)
            screenshot_time = time.time()
            screenshot_img = self.capture_left_monitor()
            screenshot_file = os.path.join(self.data_dir, f"step_{self.current_scenario_step+1}_screen.png")
            screenshot_img.save(screenshot_file)
            self.logger.info(f"단계 {self.current_scenario_step+1} 스크린샷 저장: {screenshot_file}")

            # 2. 구간별 시선 데이터 분리
            if hasattr(self, 'last_step_time'):
                t_start = self.last_step_time
            else:
                t_start = self.gaze_data[0]['timestamp'] if self.gaze_data else screenshot_time
            t_end = screenshot_time
            step_gaze = [g for g in self.gaze_data if t_start <= g['timestamp'] < t_end]
            gaze_file = os.path.join(self.data_dir, f"step_{self.current_scenario_step+1}_gaze.json")
            with open(gaze_file, 'w', encoding='utf-8') as f:
                json.dump(step_gaze, f, ensure_ascii=False, indent=2)
            self.logger.info(f"단계 {self.current_scenario_step+1} 시선 데이터 저장: {gaze_file}")

            # 3. 머문 시간 저장
            duration = t_end - t_start
            duration_file = os.path.join(self.data_dir, f"step_{self.current_scenario_step+1}_duration.txt")
            with open(duration_file, 'w', encoding='utf-8') as f:
                f.write(str(duration))
            self.logger.info(f"단계 {self.current_scenario_step+1} 머문 시간: {duration:.2f}초")

            # 4. 다음 단계로 진행
            self.last_step_time = screenshot_time
            self.create_enhanced_heatmap(self.current_scenario_step)
            self.current_scenario_step += 1
            self.logger.info(f"수동으로 다음 단계로 진행: {self.current_scenario_step}")
            if self.current_scenario_step < len(self.scenario_steps):
                next_step_info = self.scenario_steps[self.current_scenario_step]
                self.logger.info(f"=== 단계 {next_step_info['step'] + 1}: {next_step_info['description']} ===")
                self.logger.info(f"대상: {next_step_info['target_element']} (난이도: {next_step_info['difficulty']})")
    
    def create_enhanced_heatmap(self, step_number: int):
        """향상된 히트맵 생성 (왼쪽 모니터 스크린샷 + 시선 데이터)"""
        try:
            if step_number in self.step_visualizations_created:
                return
            
            # 단계별 데이터 가져오기
            step_gaze_data = self.step_gaze_data.get(step_number, [])
            step_click_events = self.step_click_events.get(step_number, [])
            
            # 단계 정보
            if step_number < len(self.scenario_steps):
                step_info = self.scenario_steps[step_number]
                actual_step_number = step_info['step'] + 1
            else:
                actual_step_number = step_number + 1
            
            # step별 스크린샷 파일 경로
            screenshot_file = os.path.join(self.data_dir, f"step_{actual_step_number}_screen.png")
            if not os.path.exists(screenshot_file):
                self.logger.warning(f"단계 {actual_step_number}의 스크린샷 파일이 없습니다: {screenshot_file}")
                return
            img = Image.open(screenshot_file)
            img_array = np.array(img)
            height, width = img_array.shape[0], img_array.shape[1]
            
            # matplotlib 설정
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
            
            # 히트맵 생성
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # 스크린샷을 배경으로 표시 (좌표계 명확히)
            ax.imshow(img_array, extent=[0, width, 0, height], origin='lower', alpha=1.0, zorder=0)
            
            # 시선 데이터가 있으면 히트맵 오버레이
            if step_gaze_data:
                x_coords = [data['x'] for data in step_gaze_data]
                y_coords = [data['y'] for data in step_gaze_data]
                h = ax.hist2d(x_coords, y_coords, bins=30, cmap='hot', alpha=0.6, zorder=1, range=[[0, width], [0, height]])
                if step_click_events:
                    click_x = [event['x'] for event in step_click_events]
                    click_y = [event['y'] for event in step_click_events]
                    ax.scatter(click_x, click_y, c='red', s=200, marker='*', label='클릭 위치', zorder=2)
                    ax.legend()
                plt.colorbar(h[3], ax=ax, label='시선 빈도')
            ax.set_title(f'단계 {actual_step_number} 시선 히트맵:\n(스크린샷 + 시선 데이터)', fontsize=14)
            ax.set_xlabel('X 좌표 (픽셀)')
            ax.set_ylabel('Y 좌표 (픽셀)')
            heatmap_file = os.path.join(self.data_dir, f"step_{actual_step_number}_enhanced_heatmap.png")
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            screenshot_file = os.path.join(self.data_dir, f"step_{actual_step_number}_screenshot.png")
            img.save(screenshot_file)
            self.logger.info(f"단계 {actual_step_number} 향상된 히트맵 생성 완료")
            print(f"📊 단계 {actual_step_number} 향상된 히트맵 생성 완료")
        except Exception as e:
            self.logger.error(f"향상된 히트맵 생성 실패: {e}")
    
    def run_scenario_collection(self):
        """향상된 시나리오 수집 실행"""
        try:
            self.logger.info("=== 향상된 키오스크 시나리오 수집 시작 ===")
            
            # 스크린샷 캡처 시작
            self.start_screenshot_capture()
            
            # 시선 추적 초기화
            if EYEWARE_AVAILABLE:
                self.initialize_eye_tracker()
                if self.tracker:
                    self.start_gaze_tracking()
                    self.logger.info("시선 추적 시작")
            
            # 웹서버 시작
            self.start_web_server()
            
            # 브라우저에서 키오스크 UI 열기
            kiosk_url = f"http://localhost:{self.server_port}"
            self.logger.info(f"키오스크 UI 열기: {kiosk_url}")
            webbrowser.open(kiosk_url)
            
            # 시나리오 진행
            self.track_scenario_progress()
            
        except Exception as e:
            self.logger.error(f"시나리오 수집 실행 실패: {e}")
        finally:
            self.cleanup()
    
    def initialize_eye_tracker(self):
        """시선 추적기 초기화"""
        if not EYEWARE_AVAILABLE:
            self.logger.warning("시선 추적을 사용할 수 없습니다.")
            return False
        
        try:
            # 뷰포트 설정
            viewport = bet.ViewportGeometry()
            viewport.point_00.x = 0
            viewport.point_00.y = 0
            viewport.point_11.x = self.screen_width
            viewport.point_11.y = self.screen_height
            
            # API 객체 생성
            self.tracker = bet.API("enhanced-kiosk-collector", viewport)
            self.logger.info("시선 추적기 초기화 성공!")
            
            # Beam 트래커 자동 시작
            self.tracker.attempt_starting_the_beam_eye_tracker()
            self.logger.info("Beam 트래커 자동 시작 요청 완료")
            
            return True
            
        except Exception as e:
            self.logger.error(f"시선 추적기 초기화 실패: {e}")
            return False
    
    def start_gaze_tracking(self):
        """시선 추적 시작"""
        if not self.tracker:
            return
        
        try:
            self.tracking_active = True
            self.gaze_thread = threading.Thread(target=self._gaze_tracking_loop)
            self.gaze_thread.daemon = True
            self.gaze_thread.start()
            self.logger.info("시선 추적이 시작되었습니다.")
        except Exception as e:
            self.logger.error(f"시선 추적 시작 실패: {e}")
    
    def _gaze_tracking_loop(self):
        """시선 추적 루프"""
        last_update_timestamp = bet.NULL_DATA_TIMESTAMP()
        error_count = 0
        
        while self.tracking_active:
            try:
                # 새로운 데이터 대기
                if self.tracker.wait_for_new_tracking_state_set(last_update_timestamp, 100):
                    tracking_state_set = self.tracker.get_latest_tracking_state_set()
                    user_state = tracking_state_set.user_state()
                    
                    # 유효한 시선 데이터 수집
                    if (user_state.unified_screen_gaze.confidence != bet.TrackingConfidence.LOST_TRACKING and
                        user_state.timestamp_in_seconds != bet.NULL_DATA_TIMESTAMP()):
                        
                        gaze = user_state.unified_screen_gaze.point_of_regard
                        gaze_data_point = {
                            'timestamp': time.time(),
                            'x': gaze.x,
                            'y': gaze.y,
                            'confidence': user_state.unified_screen_gaze.confidence,
                            'step': self.current_scenario_step
                        }
                        
                        # 전체 시선 데이터에 추가
                        self.gaze_data.append(gaze_data_point)
                        
                        # 단계별 시선 데이터에 추가
                        if self.current_scenario_step not in self.step_gaze_data:
                            self.step_gaze_data[self.current_scenario_step] = []
                        self.step_gaze_data[self.current_scenario_step].append(gaze_data_point)
                    
                    last_update_timestamp = user_state.timestamp_in_seconds
                    error_count = 0
                
                time.sleep(0.01)  # 10ms 간격
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    self.logger.error(f"시선 추적 루프 오류 (횟수: {error_count}): {e}")
                time.sleep(0.1)
    
    def start_web_server(self):
        """웹서버 시작"""
        try:
            class CustomHandler(SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    current_dir = os.getcwd()
                    self.kiosk_web_dir = os.path.join(os.path.dirname(current_dir), "kiosk web")
                    
                    build_dir = os.path.join(self.kiosk_web_dir, "build")
                    if os.path.exists(build_dir):
                        self.serve_dir = build_dir
                        print(f"✅ React 빌드 폴더 서빙: {build_dir}")
                    else:
                        self.serve_dir = self.kiosk_web_dir
                        print(f"⚠️ 기본 폴더 서빙: {self.kiosk_web_dir}")
                    
                    super().__init__(*args, directory=self.serve_dir, **kwargs)
                
                def end_headers(self):
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                    super().end_headers()
                
                def do_GET(self):
                    if self.path == '/' or self.path == '/index.html':
                        self.path = '/index.html'
                    elif not os.path.exists(os.path.join(self.serve_dir, self.path.lstrip('/'))):
                        self.path = '/index.html'
                    
                    super().do_GET()
            
            # 서버 시작
            self.server = HTTPServer(('localhost', self.server_port), CustomHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            self.logger.info(f"키오스크 UI 서버 시작: http://localhost:{self.server_port}")
            
        except Exception as e:
            self.logger.error(f"웹서버 시작 실패: {e}")
    
    def track_scenario_progress(self):
        """시나리오 진행 상태 추적"""
        try:
            self.logger.info("시나리오 진행 추적 시작")
            
            # 키보드 단축키 설정
            if KEYBOARD_AVAILABLE:
                keyboard.add_hotkey('ctrl+shift+s', self.advance_scenario_step)
                keyboard.add_hotkey('ctrl+shift+f', self.record_failure)
                keyboard.add_hotkey('ctrl+shift+r', self.reset_scenario)
                self.logger.info("키보드 단축키 등록 완료")
            
            # 시나리오 시작 안내
            self.show_scenario_instructions()
            
            # 자동 진행 모니터링
            while self.current_scenario_step < len(self.scenario_steps):
                current_step_info = self.scenario_steps[self.current_scenario_step]
                
                self.logger.info(f"=== 단계 {current_step_info['step'] + 1}: {current_step_info['description']} ===")
                self.logger.info(f"대상: {current_step_info['target_element']} (난이도: {current_step_info['difficulty']})")
                
                # 사용자 입력 대기
                self.wait_for_step_completion(current_step_info)
                
                # 다음 단계로 진행
                self.current_scenario_step += 1
                time.sleep(2)
            
            # 시나리오 완료
            self.logger.info("=== 시나리오 완료 ===")
            
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 시나리오 중단됨")
        except Exception as e:
            self.logger.error(f"진행 추적 실패: {e}")
    
    def wait_for_step_completion(self, step_info):
        """단계 완료 대기"""
        try:
            step_start_time = time.time()
            max_wait_time = 300  # 5분 대기
            
            self.logger.info(f"단계 '{step_info['target_element']}' 완료 대기 중...")
            print(f"\n⏳ 단계 {step_info['step'] + 1} 완료 대기 중...")
            print("   • Ctrl+Shift+S: 다음 단계로 진행")
            print("   • Ctrl+Shift+F: 현재 단계 실패 기록")
            print("   • Ctrl+Shift+R: 시나리오 재시작")
            
            # 자동 진행 감지 또는 수동 입력 대기
            while time.time() - step_start_time < max_wait_time:
                if KEYBOARD_AVAILABLE:
                    if keyboard.is_pressed('ctrl+shift+s'):
                        self.logger.info("수동으로 다음 단계 진행")
                        print("✅ 다음 단계로 진행합니다!")
                        break
                    elif keyboard.is_pressed('ctrl+shift+f'):
                        self.record_step_failure(step_info, "사용자 수동 실패 기록")
                        print("❌ 현재 단계를 실패로 기록하고 다음 단계로 진행합니다!")
                        break
                    elif keyboard.is_pressed('ctrl+shift+r'):
                        self.reset_scenario()
                        print("🔄 시나리오를 재시작합니다!")
                        return
                
                time.sleep(0.1)
            
            # 시간 초과 시 실패로 기록
            if time.time() - step_start_time >= max_wait_time:
                self.record_step_failure(step_info, "시간 초과")
                print("⏰ 시간 초과! 다음 단계로 진행합니다.")
            
        except Exception as e:
            self.logger.error(f"단계 완료 대기 실패: {e}")
    
    def record_step_failure(self, step_info, failure_reason):
        """단계 실패 기록"""
        try:
            failure_data = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'step': step_info['step'],
                'target_element': step_info['target_element'],
                'category': step_info['category'],
                'difficulty': step_info['difficulty'],
                'failure_reason': failure_reason,
                'gaze_data_count': len(self.gaze_data),
                'click_events_count': len(self.click_events)
            }
            
            self.error_logs.append(failure_data)
            self.logger.error(f"단계 {step_info['step'] + 1} 실패: {failure_reason}")
            
        except Exception as e:
            self.logger.error(f"실패 기록 실패: {e}")
    
    def record_failure(self):
        """현재 단계 실패 기록"""
        if self.current_scenario_step < len(self.scenario_steps):
            current_step = self.scenario_steps[self.current_scenario_step]
            self.record_step_failure(current_step, "사용자 수동 실패 기록")
    
    def reset_scenario(self):
        """시나리오 재시작"""
        self.current_scenario_step = 0
        self.gaze_data = []
        self.click_events = []
        self.error_logs = []
        self.logger.info("시나리오 재시작")
    
    def show_scenario_instructions(self):
        """시나리오 지시사항 표시"""
        print("\n" + "="*80)
        print("🔥 향상된 키오스크 시나리오 데이터 수집")
        print("="*80)
        print(f"📁 데이터 저장 위치: {self.data_dir}")
        print(f"🌐 키오스크 UI: http://localhost:{self.server_port}")
        print("\n📋 시나리오 개요:")
        print(f"   • 총 {len(self.scenario_steps)}개 단계")
        print(f"   • 스크린샷 + 시선 데이터 동기화")
        print(f"   • 향상된 히트맵 생성")
        print("\n⌨️  키보드 단축키:")
        print("   • Ctrl+Shift+S: 다음 단계로 진행")
        print("   • Ctrl+Shift+F: 현재 단계 실패 기록")
        print("   • Ctrl+Shift+R: 시나리오 재시작")
        print("="*80)
        print("\n🚀 시나리오를 시작합니다...")
        print("키오스크 UI에서 지시에 따라 메뉴를 선택해주세요.")
        print("="*80)
    
    def cleanup(self):
        """수집 후 정리"""
        self.logger.info("수집 후 정리 시작")
        
        # 데이터 수집 종료
        self.stop_screenshot_capture()
        self.stop_gaze_tracking()
        
        if hasattr(self, 'server'):
            try:
                self.server.shutdown()
                self.logger.info("웹서버 종료")
            except Exception as e:
                self.logger.error(f"웹서버 종료 실패: {e}")
        
        # 데이터 저장
        self.save_collected_data()
        
        self.logger.info("수집 후 정리 완료")
    
    def stop_gaze_tracking(self):
        """시선 추적 중지"""
        try:
            self.tracking_active = False
            if hasattr(self, 'gaze_thread'):
                self.gaze_thread.join(timeout=1.0)
            self.logger.info("시선 추적이 중지되었습니다.")
        except Exception as e:
            self.logger.error(f"시선 추적 중지 실패: {e}")
    
    def save_collected_data(self):
        """수집된 데이터 저장"""
        self.logger.info("수집된 데이터를 저장하고 있습니다...")
        print("\n수집된 데이터를 저장하고 있습니다...")
        
        try:
            # 데이터 수집 현황 출력
            self.logger.info(f"수집된 시선 데이터: {len(self.gaze_data)}개")
            self.logger.info(f"수집된 스크린샷: {len(self.screenshot_data)}개")
            self.logger.info(f"수집된 클릭 이벤트: {len(self.click_events)}개")
            self.logger.info(f"수집된 오류 로그: {len(self.error_logs)}개")
            
            print(f"수집된 시선 데이터: {len(self.gaze_data)}개")
            print(f"수집된 스크린샷: {len(self.screenshot_data)}개")
            print(f"수집된 클릭 이벤트: {len(self.click_events)}개")
            print(f"수집된 오류 로그: {len(self.error_logs)}개")
            
            # 시선 데이터 저장
            if self.gaze_data:
                gaze_file = os.path.join(self.data_dir, "gaze_data.json")
                with open(gaze_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'scenario': self.current_scenario,
                        'timestamp': self.timestamp,
                        'screen_resolution': {'width': self.screen_width, 'height': self.screen_height},
                        'gaze_data': self.gaze_data,
                        'step_gaze_data': self.step_gaze_data
                    }, f, indent=2, ensure_ascii=False)
                self.logger.info(f"시선 데이터 저장 완료: {gaze_file}")
                print(f"시선 데이터 저장: {gaze_file}")
            
            # 스크린샷 데이터 저장 (Base64 제외하고 메타데이터만)
            if self.screenshot_data:
                screenshot_meta = []
                for screenshot in self.screenshot_data:
                    meta = {
                        'timestamp': screenshot['timestamp'],
                        'datetime': screenshot['datetime'],
                        'step': screenshot['step'],
                        'screen_resolution': screenshot['screen_resolution']
                    }
                    screenshot_meta.append(meta)
                
                screenshot_file = os.path.join(self.data_dir, "screenshot_metadata.json")
                with open(screenshot_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'scenario': self.current_scenario,
                        'timestamp': self.timestamp,
                        'screenshot_count': len(self.screenshot_data),
                        'screenshot_metadata': screenshot_meta
                    }, f, indent=2, ensure_ascii=False)
                self.logger.info(f"스크린샷 메타데이터 저장 완료: {screenshot_file}")
                print(f"스크린샷 메타데이터 저장: {screenshot_file}")
            
            # 클릭 이벤트 저장
            if self.click_events:
                click_file = os.path.join(self.data_dir, "click_events.json")
                with open(click_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'scenario': self.current_scenario,
                        'timestamp': self.timestamp,
                        'click_events': self.click_events,
                        'step_click_events': self.step_click_events
                    }, f, indent=2, ensure_ascii=False)
                self.logger.info(f"클릭 이벤트 저장 완료: {click_file}")
                print(f"클릭 이벤트 저장: {click_file}")
            
            # 오류 로그 저장
            if self.error_logs:
                error_file = os.path.join(self.data_dir, "error_logs.json")
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'scenario': self.current_scenario,
                        'timestamp': self.timestamp,
                        'error_logs': self.error_logs
                    }, f, indent=2, ensure_ascii=False)
                self.logger.info(f"오류 로그 저장 완료: {error_file}")
                print(f"오류 로그 저장: {error_file}")
            
            self.logger.info(f"모든 데이터가 {self.data_dir} 디렉토리에 저장되었습니다.")
            print(f"\n모든 데이터가 {self.data_dir} 디렉토리에 저장되었습니다.")
            
        except Exception as e:
            self.logger.error(f"데이터 저장 실패: {e}")
            print(f"데이터 저장 중 오류 발생: {e}")

def main():
    """향상된 키오스크 시나리오 수집 메인 함수"""
    print("🔥 향상된 키오스크 시나리오 데이터 수집기")
    print("스크린샷과 시선 데이터를 시간과 함께 동기화")
    print("="*60)
    
    try:
        # 수집기 생성
        collector = EnhancedKioskScenarioCollector("R1_kiosk_hell_enhanced")
        
        # 시나리오 수집 실행
        collector.run_scenario_collection()
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {e}")
        print("상세 오류 정보는 로그 파일을 확인하세요.")
    finally:
        print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    main() 