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

class KioskScenarioCollector:
    def __init__(self, scenario_name: str = "kiosk_scenario"):
        """
        키오스크 시나리오 데이터 수집기 초기화 (헬 난이도 UI 버전)
        
        Args:
            scenario_name: 시나리오 이름 (로그 파일명에 사용)
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
        
        # 오류 로그
        self.error_logs = []
        
        # 실시간 시나리오 표시 관련
        self.scenario_display_active = False
        self.scenario_display_thread = None
        
        # 단계별 시각화 관련
        self.step_visualizations_created = set()  # 이미 생성된 단계 시각화 추적
        
        # 스크린샷 캡처 관련
        self.screenshot_data = []  # 스크린샷 데이터 저장
        self.screenshot_interval = 2.0  # 2초마다 스크린샷 캡처
        self.screenshot_active = False
        self.screenshot_thread = None
        
        # 헬 난이도 시나리오 정의 (복잡한 카테고리 구조)
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
                },
                {
                    "step": 3,
                    "description": "티·에이드 카테고리에서 '샤인머스캣 에이드' 선택",
                    "target_element": "샤인머스캣 에이드",
                    "expected_action": "click",
                    "category": "tea",
                    "item_id": "ade_mus",
                    "difficulty": "hard"
                },
                {
                    "step": 4,
                    "description": "디저트 카테고리에서 '뉴욕 치즈케이크' 선택",
                    "target_element": "뉴욕 치즈케이크",
                    "expected_action": "click",
                    "category": "dessert",
                    "item_id": "des_ccake",
                    "difficulty": "medium"
                },
                {
                    "step": 5,
                    "description": "베이커리·샌드위치 카테고리에서 '버터 크루아상' 선택",
                    "target_element": "버터 크루아상",
                    "expected_action": "click",
                    "category": "bakery",
                    "item_id": "bak_crois",
                    "difficulty": "hard"
                },
                {
                    "step": 6,
                    "description": "프라푸치노·블렌디드 카테고리에서 '자바칩 프라푸치노' 선택",
                    "target_element": "자바칩 프라푸치노",
                    "expected_action": "click",
                    "category": "frapp",
                    "item_id": "frp_java",
                    "difficulty": "hard"
                },
                {
                    "step": 7,
                    "description": "스무디·주스 카테고리에서 '블루베리 요거트 스무디' 선택",
                    "target_element": "블루베리 요거트 스무디",
                    "expected_action": "click",
                    "category": "smoothie",
                    "item_id": "sm_blue",
                    "difficulty": "hard"
                },
                {
                    "step": 8,
                    "description": "빙수 카테고리에서 '딸기 빙수' 선택",
                    "target_element": "딸기 빙수",
                    "expected_action": "click",
                    "category": "shavedice",
                    "item_id": "ice_straw",
                    "difficulty": "very_hard"
                },
                {
                    "step": 9,
                    "description": "시즌 스페셜 카테고리에서 '펌킨 스파이스 라떼' 선택",
                    "target_element": "펌킨 스파이스 라떼",
                    "expected_action": "click",
                    "category": "season",
                    "item_id": "ss_pump",
                    "difficulty": "very_hard"
                },
                {
                    "step": 10,
                    "description": "장바구니 보기 버튼 클릭",
                    "target_element": "장바구니 보기",
                    "expected_action": "click",
                    "category": "cart",
                    "difficulty": "easy"
                },
                {
                    "step": 11,
                    "description": "결제하기 버튼 클릭",
                    "target_element": "결제하기",
                    "expected_action": "click",
                    "category": "payment",
                    "difficulty": "medium"
                },
                {
                    "step": 12,
                    "description": "신용카드 결제 선택",
                    "target_element": "신용카드",
                    "expected_action": "click",
                    "category": "payment_method",
                    "difficulty": "medium"
                },
                {
                    "step": 13,
                    "description": "매장 식사 선택",
                    "target_element": "매장",
                    "expected_action": "click",
                    "category": "dining_type",
                    "difficulty": "easy"
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
        
        self.logger.info(f"헬 난이도 키오스크 시나리오 수집기 초기화 완료: {scenario_name}")
        
    def setup_logging(self):
        """로깅 시스템 설정"""
        log_file = os.path.join(self.data_dir, "collection.log")
        
        # 로거 설정
        self.logger = logging.getLogger('KioskCollector')
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
    
    def log_error(self, error_type: str, error_msg: str, error_details: str = ""):
        """오류 로그 기록"""
        error_log = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'type': error_type,
            'message': error_msg,
            'details': error_details,
            'step': self.current_scenario_step
        }
        self.error_logs.append(error_log)
        self.logger.error(f"[{error_type}] {error_msg}")
        if error_details:
            self.logger.debug(f"상세 정보: {error_details}")
    
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
            self.log_error("SCREEN_RESOLUTION", f"화면 해상도 감지 실패: {e}", traceback.format_exc())
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
            self.log_error("PORT_FIND", f"포트 찾기 실패: {e}", traceback.format_exc())
            return 8080  # 기본 포트
    
    def start_web_server(self):
        """웹서버 시작 (헬 난이도 UI 제공)"""
        try:
            class CustomHandler(SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    # kiosk web 폴더를 루트로 설정 (상위 디렉토리에서 찾기)
                    current_dir = os.getcwd()
                    self.kiosk_web_dir = os.path.join(os.path.dirname(current_dir), "kiosk web")
                    
                    # build 폴더가 있으면 build 폴더를 서빙, 없으면 public 폴더 서빙
                    build_dir = os.path.join(self.kiosk_web_dir, "build")
                    public_dir = os.path.join(self.kiosk_web_dir, "public")
                    
                    if os.path.exists(build_dir):
                        self.serve_dir = build_dir
                        print(f"✅ React 빌드 폴더 서빙: {build_dir}")
                    elif os.path.exists(public_dir):
                        self.serve_dir = public_dir
                        print(f"⚠️  React public 폴더 서빙: {public_dir}")
                    else:
                        self.serve_dir = self.kiosk_web_dir
                        print(f"⚠️  기본 폴더 서빙: {self.kiosk_web_dir}")
                    
                    super().__init__(*args, directory=self.serve_dir, **kwargs)
                
                def end_headers(self):
                    # CORS 헤더 추가
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                    super().end_headers()
                
                def log_message(self, format, *args):
                    # 웹서버 로그를 수집기 로거로 전달
                    pass  # 로그 비활성화
                
                def do_GET(self):
                    # React 앱의 경우 모든 경로를 index.html로 리다이렉트
                    if self.path == '/' or self.path == '/index.html':
                        self.path = '/index.html'
                    elif not os.path.exists(os.path.join(self.serve_dir, self.path.lstrip('/'))):
                        # 파일이 없으면 index.html로 리다이렉트 (React Router 지원)
                        self.path = '/index.html'
                    
                    super().do_GET()
            
            # 서버 시작
            self.server = HTTPServer(('localhost', self.server_port), CustomHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            self.logger.info(f"헬 난이도 키오스크 UI 서버 시작: http://localhost:{self.server_port}")
            
        except Exception as e:
            self.log_error("WEB_SERVER", f"웹서버 시작 실패: {e}", traceback.format_exc())
    
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
            self.tracker = bet.API("kiosk-scenario-collector", viewport)
            self.logger.info("시선 추적기 초기화 성공!")
            print("시선 추적기 초기화 성공!")
            
            # Beam 트래커 자동 시작
            self.tracker.attempt_starting_the_beam_eye_tracker()
            self.logger.info("Beam 트래커 자동 시작 요청 완료")
            
            return True
            
        except Exception as e:
            self.log_error("EYE_TRACKER", f"시선 추적기 초기화 실패: {e}", traceback.format_exc())
            print(f"시선 추적기 초기화 실패: {e}")
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
            print("시선 추적이 시작되었습니다.")
        except Exception as e:
            self.log_error("GAZE_TRACKING", f"시선 추적 시작 실패: {e}", traceback.format_exc())
    
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
                    error_count = 0  # 성공시 오류 카운트 리셋
                
                time.sleep(0.01)  # 10ms 간격
                
            except Exception as e:
                error_count += 1
                if error_count <= 5:  # 처음 5번만 로그
                    self.log_error("GAZE_LOOP", f"시선 추적 루프 오류 (횟수: {error_count}): {e}", traceback.format_exc())
                time.sleep(0.1)
    
    def stop_gaze_tracking(self):
        """시선 추적 중지"""
        try:
            self.tracking_active = False
            if hasattr(self, 'gaze_thread'):
                self.gaze_thread.join(timeout=1.0)
            self.logger.info("시선 추적이 중지되었습니다.")
        except Exception as e:
            self.log_error("GAZE_STOP", f"시선 추적 중지 실패: {e}", traceback.format_exc())
    
    def record_click_event(self, x: int, y: int, element_text: str = "", success: bool = True):
        """클릭 이벤트 기록"""
        try:
            click_event = {
                'timestamp': time.time(),
                'x': x,
                'y': y,
                'element_text': element_text,
                'step': self.current_scenario_step,
                'success': success,
                'scenario_step': self.scenario_steps[self.current_scenario_step] if self.current_scenario_step < len(self.scenario_steps) else None
            }
            
            # 전체 클릭 이벤트에 추가
            self.click_events.append(click_event)
            
            # 단계별 클릭 이벤트에 추가
            if self.current_scenario_step not in self.step_click_events:
                self.step_click_events[self.current_scenario_step] = []
            self.step_click_events[self.current_scenario_step].append(click_event)
            
            self.logger.info(f"클릭 이벤트 기록: ({x}, {y}) - {element_text}")
            print(f"클릭 이벤트 기록: ({x}, {y}) - {element_text}")
        except Exception as e:
            self.log_error("CLICK_RECORD", f"클릭 이벤트 기록 실패: {e}", traceback.format_exc())
    
    def show_scenario_instructions(self):
        """헬 난이도 시나리오 지시사항 표시"""
        print("\n" + "="*80)
        print("🔥 헬 난이도 키오스크 시나리오 데이터 수집")
        print("="*80)
        print(f"📁 데이터 저장 위치: {self.data_dir}")
        print(f"🌐 키오스크 UI: http://localhost:{self.server_port}")
        print(f"📊 진행 상황: http://localhost:{self.scenario_display_port}")
        print("\n📋 시나리오 개요:")
        print(f"   • 총 {len(self.scenario_steps)}개 단계")
        print(f"   • 복잡한 카테고리 구조 (10개 카테고리)")
        print(f"   • 다양한 난이도 (easy, medium, hard, very_hard)")
        print("\n🎯 주요 목표:")
        print("   • 고령자의 복잡한 UI 탐색 패턴 분석")
        print("   • 카테고리 간 이동 시 시선 패턴 연구")
        print("   • 실패 지점과 시선 데이터의 상관관계 분석")
        print("\n⌨️  키보드 단축키:")
        print("   • Ctrl+Shift+S: 다음 단계로 진행")
        print("   • Ctrl+Shift+F: 현재 단계 실패 기록")
        print("   • Ctrl+Shift+R: 시나리오 재시작")
        print("\n⚠️  주의사항:")
        print("   • 시선 추적이 활성화되어 있습니다")
        print("   • 모든 클릭과 시선 움직임이 기록됩니다")
        print("   • 실패한 지점은 자동으로 분석됩니다")
        print("="*80)
        print("\n🚀 시나리오를 시작합니다...")
        print("키오스크 UI에서 지시에 따라 메뉴를 선택해주세요.")
        print("진행 상황은 별도 창에서 실시간으로 확인할 수 있습니다.")
        print("\n" + "="*80)
    
    def show_current_step(self):
        """현재 단계 정보 표시 (실시간 표시로 대체됨)"""
        pass  # 실시간 시나리오 표시로 대체됨
    
    def save_kiosk_ui_screenshot(self, url, save_path, width=None, height=None):
        """키오스크 UI 웹페이지를 스크린샷으로 저장 (Selenium 4.x 호환)"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
        except ImportError:
            self.logger.error("selenium 패키지가 설치되어 있지 않습니다. pip install selenium 필요")
            print("selenium 패키지가 설치되어 있지 않습니다. pip install selenium 필요")
            return
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        w = width if width else 1920
        h = height if height else 1080
        chrome_options.add_argument(f'--window-size={w},{h}')
        driver_path = os.path.abspath(os.path.join(os.getcwd(), 'chromedriver.exe'))
        service = Service(driver_path)
        try:
            driver = webdriver.Chrome(service=service, options=chrome_options)
            driver.get(url)
            import time as _time
            _time.sleep(2)  # 렌더링 대기
            driver.set_window_size(w, h)
            _time.sleep(0.5)
            driver.save_screenshot(save_path)
            self.logger.info(f"키오스크 UI 스크린샷 저장: {save_path}")
            print(f"키오스크 UI 스크린샷 저장: {save_path}")
        except Exception as e:
            self.log_error("SCREENSHOT", f"스크린샷 저장 실패: {e}")
            print(f"스크린샷 저장 실패: {e}")
        finally:
            try:
                driver.quit()
            except:
                pass

    def run_scenario_collection(self):
        """헬 난이도 시나리오 수집 실행"""
        try:
            self.logger.info("=== 헬 난이도 키오스크 시나리오 수집 시작 ===")
            
            # 시선 추적 초기화
            if EYEWARE_AVAILABLE:
                self.initialize_eye_tracker()
                if self.tracker:
                    self.start_gaze_tracking()
                    self.logger.info("시선 추적 시작")
            
            # 웹서버 시작 (헬 난이도 UI)
            self.start_web_server()
            
            # 실시간 시나리오 표시 시작
            self.start_scenario_display()
            
            # 웹서버 연결 상태 확인
            self.check_web_servers()
            
            # 브라우저에서 키오스크 UI 열기
            kiosk_url = f"http://localhost:{self.server_port}"
            self.logger.info(f"키오스크 UI 열기: {kiosk_url}")
            webbrowser.open(kiosk_url)
            
            # 시나리오 표시 창 열기
            scenario_url = f"http://localhost:{self.scenario_display_port}"
            self.logger.info(f"시나리오 진행 상황 열기: {scenario_url}")
            webbrowser.open(scenario_url)
            
            # 시나리오 진행 상태 추적
            self.track_scenario_progress()
            
        except Exception as e:
            self.log_error("SCENARIO_COLLECTION", f"시나리오 수집 실행 실패: {e}", traceback.format_exc())
        finally:
            self.cleanup()
    
    def check_web_servers(self):
        """웹서버 연결 상태 확인"""
        try:
            import requests
            
            # 키오스크 UI 서버 확인
            kiosk_url = f"http://localhost:{self.server_port}"
            try:
                response = requests.get(kiosk_url, timeout=5)
                if response.status_code == 200:
                    self.logger.info(f"✅ 키오스크 UI 서버 연결 성공: {kiosk_url}")
                    print(f"✅ 키오스크 UI 서버 연결 성공: {kiosk_url}")
                else:
                    self.logger.warning(f"⚠️ 키오스크 UI 서버 응답 오류: {response.status_code}")
                    print(f"⚠️ 키오스크 UI 서버 응답 오류: {response.status_code}")
            except Exception as e:
                self.logger.error(f"❌ 키오스크 UI 서버 연결 실패: {e}")
                print(f"❌ 키오스크 UI 서버 연결 실패: {e}")
            
            # 시나리오 표시 서버 확인
            scenario_url = f"http://localhost:{self.scenario_display_port}"
            try:
                response = requests.get(scenario_url, timeout=5)
                if response.status_code == 200:
                    self.logger.info(f"✅ 시나리오 표시 서버 연결 성공: {scenario_url}")
                    print(f"✅ 시나리오 표시 서버 연결 성공: {scenario_url}")
                else:
                    self.logger.warning(f"⚠️ 시나리오 표시 서버 응답 오류: {response.status_code}")
                    print(f"⚠️ 시나리오 표시 서버 응답 오류: {response.status_code}")
            except Exception as e:
                self.logger.error(f"❌ 시나리오 표시 서버 연결 실패: {e}")
                print(f"❌ 시나리오 표시 서버 연결 실패: {e}")
            
            # SSE 엔드포인트 확인
            sse_url = f"http://localhost:{self.scenario_display_port}/events"
            try:
                response = requests.get(sse_url, timeout=5)
                if response.status_code == 200:
                    self.logger.info(f"✅ SSE 이벤트 스트림 연결 성공: {sse_url}")
                    print(f"✅ SSE 이벤트 스트림 연결 성공: {sse_url}")
                else:
                    self.logger.warning(f"⚠️ SSE 이벤트 스트림 응답 오류: {response.status_code}")
                    print(f"⚠️ SSE 이벤트 스트림 응답 오류: {response.status_code}")
            except Exception as e:
                self.logger.error(f"❌ SSE 이벤트 스트림 연결 실패: {e}")
                print(f"❌ SSE 이벤트 스트림 연결 실패: {e}")
                
        except ImportError:
            self.logger.warning("requests 모듈이 없어 웹서버 연결 확인을 건너뜁니다.")
            print("⚠️ requests 모듈이 없어 웹서버 연결 확인을 건너뜁니다.")
        except Exception as e:
            self.log_error("WEB_SERVER_CHECK", f"웹서버 연결 확인 실패: {e}", traceback.format_exc())
            
    def track_scenario_progress(self):
        """시나리오 진행 상태 추적 (자동 진행)"""
        try:
            self.logger.info("시나리오 진행 추적 시작")
            
            # 키보드 단축키 설정
            if KEYBOARD_AVAILABLE:
                keyboard.add_hotkey('ctrl+shift+s', self.advance_scenario_step)
                keyboard.add_hotkey('ctrl+shift+f', self.record_failure)
                keyboard.add_hotkey('ctrl+shift+r', self.reset_scenario)
                self.logger.info("키보드 단축키 등록:")
                self.logger.info("  Ctrl+Shift+S: 다음 단계로 진행")
                self.logger.info("  Ctrl+Shift+F: 현재 단계 실패 기록")
                self.logger.info("  Ctrl+Shift+R: 시나리오 재시작")
            
            # 시나리오 시작 안내
            self.show_scenario_instructions()
            
            # 자동 진행 모니터링
            while self.current_scenario_step < len(self.scenario_steps):
                current_step_info = self.scenario_steps[self.current_scenario_step]
                
                self.logger.info(f"=== 단계 {current_step_info['step'] + 1}: {current_step_info['description']} ===")
                self.logger.info(f"대상: {current_step_info['target_element']} (난이도: {current_step_info['difficulty']})")
                
                # 사용자 입력 대기 (자동 진행 또는 수동 입력)
                self.wait_for_step_completion(current_step_info)
                
                # 다음 단계로 진행 (시각화는 advance_scenario_step에서 처리)
                                self.current_scenario_step += 1
                
                # 잠시 대기 (사용자가 다음 단계를 확인할 시간)
                time.sleep(2)
            
            # 시나리오 완료
            self.logger.info("=== 시나리오 완료 ===")
            self.show_completion_message()
            
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 시나리오 중단됨")
        except Exception as e:
            self.log_error("PROGRESS_TRACKING", f"진행 추적 실패: {e}", traceback.format_exc())
    
    def wait_for_step_completion(self, step_info):
        """단계 완료 대기 (수동 진행 또는 자동 진행)"""
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
                    # 키보드 입력 확인
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
                        return  # 재시작이므로 루프 종료
                
                # 잠시 대기
                            time.sleep(0.1)
            
            # 시간 초과 시 실패로 기록
            if time.time() - step_start_time >= max_wait_time:
                self.record_step_failure(step_info, "시간 초과")
                print("⏰ 시간 초과! 다음 단계로 진행합니다.")
            
                        except Exception as e:
            self.log_error("STEP_WAIT", f"단계 완료 대기 실패: {e}", traceback.format_exc())
    
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
            
            # 실패 시 시선 데이터 저장
            if self.gaze_data:
                failure_gaze_file = os.path.join(self.data_dir, f"failure_gaze_step_{step_info['step'] + 1}.json")
                with open(failure_gaze_file, 'w', encoding='utf-8') as f:
                    json.dump(self.gaze_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"실패 시 시선 데이터 저장: {failure_gaze_file}")
            
        except Exception as e:
            self.log_error("FAILURE_RECORD", f"실패 기록 실패: {e}", traceback.format_exc())
    
    def advance_scenario_step(self):
        """다음 단계로 진행"""
        # 현재 단계를 완료하고 다음 단계로 진행
        if self.current_scenario_step < len(self.scenario_steps):
            # 현재 단계의 시각화 생성
            self.create_step_visualizations(self.current_scenario_step)
            
            # 다음 단계로 진행
                        self.current_scenario_step += 1
            self.logger.info(f"수동으로 다음 단계로 진행: {self.current_scenario_step}")
            
            # 다음 단계가 있으면 해당 단계 정보 출력
            if self.current_scenario_step < len(self.scenario_steps):
                next_step_info = self.scenario_steps[self.current_scenario_step]
                self.logger.info(f"=== 단계 {next_step_info['step'] + 1}: {next_step_info['description']} ===")
                self.logger.info(f"대상: {next_step_info['target_element']} (난이도: {next_step_info['difficulty']})")
    
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

    def start_scenario_display(self):
        """실시간 시나리오 표시 서버 시작"""
        try:
            class ScenarioDisplayHandler(SimpleHTTPRequestHandler):
                def __init__(self, *args, collector=None, **kwargs):
                    self.collector = collector
                    super().__init__(*args, **kwargs)
                
                def do_GET(self):
                    if self.path == '/':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html; charset=utf-8')
                        self.end_headers()
                        
                        html_content = self.generate_scenario_html()
                        self.wfile.write(html_content.encode('utf-8'))
                    elif self.path == '/events':
                        # Server-Sent Events 엔드포인트
                        self.send_response(200)
                        self.send_header('Content-type', 'text/event-stream')
                        self.send_header('Cache-Control', 'no-cache')
                        self.send_header('Connection', 'keep-alive')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        # 실시간 이벤트 스트림
                        self.stream_events()
                    else:
                        super().do_GET()
                
                def stream_events(self):
                    """실시간 이벤트 스트림"""
                    try:
                        while True:
                            current_step = self.collector.current_scenario_step
                            total_steps = len(self.collector.scenario_steps)
                            
                            if current_step < total_steps:
                                step_info = self.collector.scenario_steps[current_step]
                                next_step_info = self.collector.scenario_steps[current_step + 1] if current_step + 1 < total_steps else None
                            else:
                                step_info = None
                                next_step_info = None
                            
                            # SSE 데이터 전송
                            data = {
                                'current_step': current_step,
                                'total_steps': total_steps,
                                'progress_percent': (current_step / total_steps) * 100 if total_steps > 0 else 0,
                                'step_info': step_info,
                                'next_step_info': next_step_info,
                                'gaze_data_count': len(self.collector.gaze_data),
                                'click_events_count': len(self.collector.click_events)
                            }
                            
                            event_data = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                            self.wfile.write(event_data.encode('utf-8'))
                            self.wfile.flush()
                            
                            time.sleep(1)  # 1초마다 업데이트
                            
                    except (BrokenPipeError, ConnectionResetError):
                        # 클라이언트 연결 종료
                        pass
        except Exception as e:
                        print(f"이벤트 스트림 오류: {e}")
                
                def generate_scenario_html(self):
                    html = """
                    <!DOCTYPE html>
                    <html lang="ko">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>키오스크 시나리오 진행 상황</title>
                        <style>
                            body {
                                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                                margin: 0;
                                padding: 20px;
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                color: white;
                                min-height: 100vh;
                            }
                            .container {
                                max-width: 800px;
                                margin: 0 auto;
                                background: rgba(255, 255, 255, 0.1);
                                border-radius: 15px;
                                padding: 30px;
                                backdrop-filter: blur(10px);
                                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                            }
                            .header {
                                text-align: center;
                                margin-bottom: 30px;
                            }
                            .header h1 {
                                margin: 0;
                                font-size: 2.5em;
                                font-weight: 300;
                                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
                            }
                            .progress-container {
                                margin: 30px 0;
                            }
                            .progress-bar {
                                width: 100%;
                                height: 20px;
                                background: rgba(255, 255, 255, 0.2);
                                border-radius: 10px;
                                overflow: hidden;
                                position: relative;
                            }
                            .progress-fill {
                                height: 100%;
                                background: linear-gradient(90deg, #4CAF50, #8BC34A);
                                transition: width 0.5s ease;
                                border-radius: 10px;
                            }
                            .progress-text {
                                text-align: center;
                                margin-top: 10px;
                                font-size: 1.2em;
                                font-weight: 500;
                            }
                            .current-step {
                                background: rgba(255, 255, 255, 0.15);
                                border-radius: 10px;
                                padding: 25px;
                                margin: 20px 0;
                                border-left: 5px solid #4CAF50;
                            }
                            .next-step {
                                background: rgba(255, 255, 255, 0.1);
                                border-radius: 10px;
                                padding: 20px;
                                margin: 20px 0;
                                border-left: 5px solid #FF9800;
                                opacity: 0.8;
                            }
                            .step-title {
                                font-size: 1.3em;
                                font-weight: 600;
                                margin-bottom: 10px;
                                color: #FFD700;
                            }
                            .step-description {
                                font-size: 1.1em;
                                line-height: 1.6;
                                margin-bottom: 15px;
                            }
                            .step-details {
                                display: grid;
                                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                                gap: 15px;
                                margin-top: 15px;
                            }
                            .detail-item {
                                background: rgba(255, 255, 255, 0.1);
                                padding: 10px;
                                border-radius: 8px;
                                text-align: center;
                            }
                            .detail-label {
                                font-size: 0.9em;
                                opacity: 0.8;
                                margin-bottom: 5px;
                            }
                            .detail-value {
                                font-size: 1.1em;
                                font-weight: 600;
                                color: #FFD700;
                            }
                            .difficulty-badge {
                                display: inline-block;
                                padding: 5px 12px;
                                border-radius: 20px;
                                font-size: 0.9em;
                                font-weight: 600;
                                text-transform: uppercase;
                            }
                            .difficulty-easy { background: #4CAF50; }
                            .difficulty-medium { background: #FF9800; }
                            .difficulty-hard { background: #F44336; }
                            .difficulty-very_hard { background: #9C27B0; }
                            .status-indicator {
                                text-align: center;
                                margin-top: 20px;
                                font-size: 0.9em;
                                opacity: 0.7;
                            }
                            .connection-status {
                                display: inline-block;
                                width: 10px;
                                height: 10px;
                                border-radius: 50%;
                                margin-right: 8px;
                            }
                            .connected { background: #4CAF50; }
                            .disconnected { background: #F44336; }
                            .data-stats {
                                background: rgba(255, 255, 255, 0.1);
                                border-radius: 10px;
                                padding: 15px;
                                margin: 20px 0;
                                text-align: center;
                            }
                        </style>
                    </head>
                    <body>
                        <div class="container">
                            <div class="header">
                                <h1>🔍 키오스크 시나리오 진행 상황</h1>
                            </div>
                            
                            <div class="progress-container">
                                <div class="progress-bar">
                                    <div class="progress-fill" id="progress-fill"></div>
                                </div>
                                <div class="progress-text" id="progress-text">
                                    연결 중...
                                </div>
                            </div>
                            
                            <div class="data-stats" id="data-stats">
                                <div>시선 데이터: <span id="gaze-count">0</span>개</div>
                                <div>클릭 이벤트: <span id="click-count">0</span>개</div>
                            </div>
                            
                            <div id="current-step-container"></div>
                            <div id="next-step-container"></div>
                            
                            <div class="status-indicator">
                                <span class="connection-status" id="connection-status"></span>
                                <span id="status-text">연결 중...</span>
                            </div>
                        </div>
                        
                        <script>
                            let eventSource;
                            
                            function connectSSE() {
                                try {
                                    eventSource = new EventSource('/events');
                                    
                                    eventSource.onopen = function() {
                                        updateConnectionStatus(true, '실시간 연결됨');
                                    };
                                    
                                    eventSource.onmessage = function(event) {
                                        const data = JSON.parse(event.data);
                                        updateUI(data);
                                    };
                                    
                                    eventSource.onerror = function() {
                                        updateConnectionStatus(false, '연결 끊어짐 - 재연결 시도 중...');
                                        eventSource.close();
                                        setTimeout(connectSSE, 2000);
                                    };
                                    
                                } catch (error) {
                                    console.error('SSE 연결 실패:', error);
                                    updateConnectionStatus(false, '연결 실패');
                                }
                            }
                            
                            function updateConnectionStatus(connected, text) {
                                const statusEl = document.getElementById('connection-status');
                                const textEl = document.getElementById('status-text');
                                
                                statusEl.className = 'connection-status ' + (connected ? 'connected' : 'disconnected');
                                textEl.textContent = text;
                            }
                            
                            function updateUI(data) {
                                // 진행률 업데이트
                                const progressFill = document.getElementById('progress-fill');
                                const progressText = document.getElementById('progress-text');
                                
                                progressFill.style.width = data.progress_percent + '%';
                                progressText.textContent = `진행률: ${data.current_step}/${data.total_steps} (${data.progress_percent.toFixed(1)}%)`;
                                
                                // 데이터 통계 업데이트
                                document.getElementById('gaze-count').textContent = data.gaze_data_count;
                                document.getElementById('click-count').textContent = data.click_events_count;
                                
                                // 현재 단계 업데이트
                                const currentStepContainer = document.getElementById('current-step-container');
                                if (data.step_info) {
                                    const difficultyClass = 'difficulty-' + (data.step_info.difficulty || 'medium');
                                    currentStepContainer.innerHTML = `
                                        <div class="current-step">
                                            <div class="step-title">🎯 현재 단계: ${data.step_info.step + 1}</div>
                                            <div class="step-description">${data.step_info.description}</div>
                                            <div class="step-details">
                                                <div class="detail-item">
                                                    <div class="detail-label">대상 요소</div>
                                                    <div class="detail-value">${data.step_info.target_element}</div>
                                                </div>
                                                <div class="detail-item">
                                                    <div class="detail-label">카테고리</div>
                                                    <div class="detail-value">${data.step_info.category}</div>
                                                </div>
                                                <div class="detail-item">
                                                    <div class="detail-label">난이도</div>
                                                    <div class="detail-value">
                                                        <span class="difficulty-badge ${difficultyClass}">
                                                            ${data.step_info.difficulty || 'medium'}
                                                        </span>
                                                    </div>
                                                </div>
                                                <div class="detail-item">
                                                    <div class="detail-label">예상 액션</div>
                                                    <div class="detail-value">${data.step_info.expected_action}</div>
                                                </div>
                                            </div>
                                        </div>
                                    `;
                                } else {
                                    currentStepContainer.innerHTML = '';
                                }
                                
                                // 다음 단계 업데이트
                                const nextStepContainer = document.getElementById('next-step-container');
                                if (data.next_step_info) {
                                    nextStepContainer.innerHTML = `
                                        <div class="next-step">
                                            <div class="step-title">⏭️ 다음 단계: ${data.next_step_info.step + 1}</div>
                                            <div class="step-description">${data.next_step_info.description}</div>
                                        </div>
                                    `;
                                } else if (data.current_step >= data.total_steps) {
                                    nextStepContainer.innerHTML = `
                                        <div class="current-step">
                                            <div class="step-title">🎉 시나리오 완료!</div>
                                            <div class="step-description">모든 단계를 성공적으로 완료했습니다.</div>
                                        </div>
                                    `;
                                } else {
                                    nextStepContainer.innerHTML = '';
                                }
                            }
                            
                            // 페이지 로드 시 SSE 연결
                            window.addEventListener('load', connectSSE);
                            
                            // 페이지 언로드 시 연결 종료
                            window.addEventListener('beforeunload', function() {
                                if (eventSource) {
                                    eventSource.close();
                                }
                            });
                        </script>
                    </body>
                    </html>
                    """
                    return html
                
                def log_message(self, format, *args):
                    # 웹서버 로그를 수집기 로거로 전달
                    if hasattr(self, 'collector') and self.collector:
                        self.collector.logger.debug(f"시나리오 표시 서버: {format % args}")
            
            # 서버 시작
            server = HTTPServer(('localhost', self.scenario_display_port), 
                              lambda *args, **kwargs: ScenarioDisplayHandler(*args, collector=self, **kwargs))
            
            self.scenario_display_thread = threading.Thread(target=server.serve_forever, daemon=True)
            self.scenario_display_thread.start()
            self.scenario_display_active = True
            
            self.logger.info(f"실시간 시나리오 표시 서버 시작: http://localhost:{self.scenario_display_port}")
            
        except Exception as e:
            self.log_error("SCENARIO_DISPLAY", f"시나리오 표시 서버 시작 실패: {e}", traceback.format_exc())
    
    def stop_scenario_display(self):
        """실시간 시나리오 표시 서버 중지"""
        if self.scenario_display_active:
            self.scenario_display_active = False
            self.logger.info("실시간 시나리오 표시 서버 중지")

    def cleanup(self):
        """수집 후 정리"""
        self.logger.info("수집 후 정리 시작")
        
            # 데이터 수집 종료
            self.stop_gaze_tracking()
            if hasattr(self, 'server'):
                try:
                    self.server.shutdown()
                    self.logger.info("웹서버 종료")
                except Exception as e:
                    self.log_error("SERVER_SHUTDOWN", f"웹서버 종료 실패: {e}", traceback.format_exc())
            
            # 데이터 저장
            self.save_collected_data()
        
        self.logger.info("수집 후 정리 완료")
    
    def save_collected_data(self):
        """수집된 데이터 저장"""
        self.logger.info("수집된 데이터를 저장하고 있습니다...")
        print("\n수집된 데이터를 저장하고 있습니다...")
        
        try:
            # 데이터 수집 현황 출력
            self.logger.info(f"수집된 시선 데이터: {len(self.gaze_data)}개")
            self.logger.info(f"수집된 클릭 이벤트: {len(self.click_events)}개")
            self.logger.info(f"수집된 오류 로그: {len(self.error_logs)}개")
            print(f"수집된 시선 데이터: {len(self.gaze_data)}개")
            print(f"수집된 클릭 이벤트: {len(self.click_events)}개")
            print(f"수집된 오류 로그: {len(self.error_logs)}개")
            
            # 시선 데이터 저장
            if self.gaze_data:
                self.logger.info("시선 데이터 저장 시작...")
                gaze_file = os.path.join(self.data_dir, "gaze_data.json")
                with open(gaze_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'scenario': self.current_scenario,
                        'timestamp': self.timestamp,
                        'screen_resolution': {'width': self.screen_width, 'height': self.screen_height},
                        'gaze_data': self.gaze_data,
                        'step_gaze_data': self.step_gaze_data  # 단계별 데이터 추가
                    }, f, indent=2, ensure_ascii=False)
                self.logger.info(f"시선 데이터 저장 완료: {gaze_file}")
                print(f"시선 데이터 저장: {gaze_file}")
            else:
                self.logger.warning("저장할 시선 데이터가 없습니다.")
                print("저장할 시선 데이터가 없습니다.")
            
            # 클릭 이벤트 저장
            if self.click_events:
                self.logger.info("클릭 이벤트 저장 시작...")
                click_file = os.path.join(self.data_dir, "click_events.json")
                with open(click_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'scenario': self.current_scenario,
                        'timestamp': self.timestamp,
                        'click_events': self.click_events,
                        'step_click_events': self.step_click_events  # 단계별 데이터 추가
                    }, f, indent=2, ensure_ascii=False)
                self.logger.info(f"클릭 이벤트 저장 완료: {click_file}")
                print(f"클릭 이벤트 저장: {click_file}")
            else:
                self.logger.warning("저장할 클릭 이벤트가 없습니다.")
                print("저장할 클릭 이벤트가 없습니다.")
            
            # 오류 로그 저장
            if self.error_logs:
                self.logger.info("오류 로그 저장 시작...")
                error_file = os.path.join(self.data_dir, "error_logs.json")
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'scenario': self.current_scenario,
                        'timestamp': self.timestamp,
                        'error_logs': self.error_logs
                    }, f, indent=2, ensure_ascii=False)
                self.logger.info(f"오류 로그 저장 완료: {error_file}")
                print(f"오류 로그 저장: {error_file}")
            else:
                self.logger.info("저장할 오류 로그가 없습니다.")
                print("저장할 오류 로그가 없습니다.")
            
            # CSV 형태로도 저장
            self.logger.info("CSV 데이터 저장 시작...")
            self.save_csv_data()
            self.logger.info("CSV 데이터 저장 완료")
            
            # 시각화 생성
            self.logger.info("시각화 생성 시작...")
            self.create_visualizations()
            self.logger.info("시각화 생성 완료")
            
            # 단계별 요약 리포트 생성
            self.create_step_summary_report()
            
            self.logger.info(f"모든 데이터가 {self.data_dir} 디렉토리에 저장되었습니다.")
            print(f"\n모든 데이터가 {self.data_dir} 디렉토리에 저장되었습니다.")
            
        except Exception as e:
            self.log_error("DATA_SAVE", f"데이터 저장 실패: {e}", traceback.format_exc())
            print(f"데이터 저장 중 오류 발생: {e}")
            print("상세 오류 정보는 로그 파일을 확인하세요.")
    
    def save_csv_data(self):
        """CSV 형태로 데이터 저장"""
        try:
            # 시선 데이터 CSV
            if self.gaze_data:
                self.logger.info("시선 데이터 CSV 저장 시작...")
                gaze_csv = os.path.join(self.data_dir, "gaze_data.csv")
                with open(gaze_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'x', 'y', 'confidence', 'step'])
                    for data in self.gaze_data:
                        writer.writerow([
                            data['timestamp'],
                            data['x'],
                            data['y'],
                            data['confidence'],
                            data['step']
                        ])
                self.logger.info(f"시선 데이터 CSV 저장 완료: {gaze_csv}")
                print(f"시선 데이터 CSV 저장: {gaze_csv}")
            
            # 클릭 이벤트 CSV
            if self.click_events:
                self.logger.info("클릭 이벤트 CSV 저장 시작...")
                click_csv = os.path.join(self.data_dir, "click_events.csv")
                with open(click_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'x', 'y', 'element_text', 'step', 'success', 'scenario_step'])
                    for event in self.click_events:
                        writer.writerow([
                            event['timestamp'],
                            event['x'],
                            event['y'],
                            event['element_text'],
                            event['step'],
                            event['success'],
                            event['scenario_step']['step'] if event['scenario_step'] else ''
                        ])
                self.logger.info(f"클릭 이벤트 CSV 저장 완료: {click_csv}")
                print(f"클릭 이벤트 CSV 저장: {click_csv}")
            
            # 오류 로그 CSV
            if self.error_logs:
                self.logger.info("오류 로그 CSV 저장 시작...")
                error_csv = os.path.join(self.data_dir, "error_logs.csv")
                with open(error_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'datetime', 'type', 'message', 'details', 'step'])
                    for error in self.error_logs:
                        writer.writerow([
                            error['timestamp'],
                            error['datetime'],
                            error['type'],
                            error['message'],
                            error['details'],
                            error['step']
                        ])
                self.logger.info(f"오류 로그 CSV 저장 완료: {error_csv}")
                print(f"오류 로그 CSV 저장: {error_csv}")
                
        except Exception as e:
            self.log_error("CSV_SAVE", f"CSV 데이터 저장 실패: {e}", traceback.format_exc())
            print(f"CSV 데이터 저장 중 오류 발생: {e}")
    
    def create_visualizations(self):
        """데이터 시각화 생성 (UI 배경 위에 히트맵 오버레이)"""
        if not self.gaze_data:
            self.logger.warning("시각화할 시선 데이터가 없습니다.")
            print("시각화할 시선 데이터가 없습니다.")
            return
        
        try:
            self.logger.info("matplotlib 설정 시작...")
            # 한글 폰트 설정
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
            self.logger.info("matplotlib 설정 완료")
            
            # 시선 데이터
            x_coords = [data['x'] for data in self.gaze_data]
            y_coords = [data['y'] for data in self.gaze_data]
            
            # 배경 UI 이미지 경로
            ui_img_path = os.path.join(self.data_dir, 'kiosk_ui.png')
            has_bg = os.path.exists(ui_img_path)
            
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
            if has_bg:
                ui_bg = plt.imread(ui_img_path)
                height, width = ui_bg.shape[0], ui_bg.shape[1]
                ax.imshow(ui_bg, extent=[0, width, 0, height])
                # 히트맵 오버레이 (alpha=0.5)
                h = ax.hist2d(x_coords, y_coords, bins=50, cmap='hot', alpha=0.5)
                ax.set_xlim(0, width)
                ax.set_ylim(0, height)
            else:
                h = plt.hist2d(x_coords, y_coords, bins=50, cmap='hot')
            
            plt.colorbar(h[3], ax=ax, label='시선 빈도')
            plt.xlabel('X 좌표 (픽셀)')
            plt.ylabel('Y 좌표 (픽셀)')
            plt.title(f'키오스크 시선 추적 히트맵 (UI 오버레이)')
            
            # 클릭 위치 표시
            if self.click_events:
                click_x = [event['x'] for event in self.click_events]
                click_y = [event['y'] for event in self.click_events]
                plt.scatter(click_x, click_y, c='red', s=100, marker='*', label='클릭 위치')
                plt.legend()
            
            plt.tight_layout()
            heatmap_file = os.path.join(self.data_dir, "gaze_heatmap.png")
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"시선 히트맵 저장 완료: {heatmap_file}")
            print(f"시선 히트맵 저장: {heatmap_file}")
            
            # 시선 궤적 시각화 (기존과 동일)
            if len(self.gaze_data) > 1:
                self.logger.info("시선 궤적 생성 시작...")
                plt.figure(figsize=(12, 8))
                plt.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=0.5)
                plt.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', label='시작')
                plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='o', label='끝')
                if self.click_events:
                    click_x = [event['x'] for event in self.click_events]
                    click_y = [event['y'] for event in self.click_events]
                    plt.scatter(click_x, click_y, c='red', s=100, marker='*', label='클릭')
                plt.xlabel('X 좌표 (픽셀)')
                plt.ylabel('Y 좌표 (픽셀)')
                plt.title('시선 이동 궤적')
                plt.legend()
                plt.grid(True, alpha=0.3)
                trajectory_file = os.path.join(self.data_dir, "gaze_trajectory.png")
                plt.savefig(trajectory_file, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"시선 궤적 저장 완료: {trajectory_file}")
                print(f"시선 궤적 저장: {trajectory_file}")
        except Exception as e:
            self.log_error("VISUALIZATION", f"시각화 생성 실패: {e}", traceback.format_exc())
            print(f"시각화 생성 중 오류 발생: {e}")
            print("상세 오류 정보는 로그 파일을 확인하세요.")

    def show_completion_message(self):
        """시나리오 완료 메시지 표시"""
        print("\n모든 시나리오가 성공적으로 완료되었습니다!")
        self.logger.info("모든 시나리오가 성공적으로 완료되었습니다!")

    def create_step_visualizations(self, step_number: int):
        """특정 단계의 시각화 생성"""
        try:
            # 이미 생성된 단계인지 확인
            if step_number in self.step_visualizations_created:
                return
            
            # 단계별 시선 데이터 확인 (데이터가 없어도 시각화 생성)
            step_gaze_data = self.step_gaze_data.get(step_number, [])
            step_click_events = self.step_click_events.get(step_number, [])
            
            # 단계 정보 가져오기 (step_number는 배열 인덱스, 실제 step 번호는 +1)
            if step_number < len(self.scenario_steps):
                step_info = self.scenario_steps[step_number]
                actual_step_number = step_info['step'] + 1  # 실제 step 번호 (1, 2, 3, ...)
                step_title = f"단계{actual_step_number}_{step_info['target_element']}"
            else:
                step_title = f"단계{step_number}"
                actual_step_number = step_number + 1
            
            # matplotlib 설정
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
            
            # 시선 데이터가 있는 경우에만 히트맵과 궤적 생성
            if step_gaze_data:
                # 시선 데이터 추출
                x_coords = [data['x'] for data in step_gaze_data]
                y_coords = [data['y'] for data in step_gaze_data]
                
                # 1. 히트맵 생성
                plt.figure(figsize=(12, 8))
                ax = plt.gca()
                
                # 히트맵 생성
                h = plt.hist2d(x_coords, y_coords, bins=30, cmap='hot', alpha=0.7)
                
                # 클릭 위치 표시
                if step_click_events:
                    click_x = [event['x'] for event in step_click_events]
                    click_y = [event['y'] for event in step_click_events]
                    plt.scatter(click_x, click_y, c='red', s=150, marker='*', label='클릭 위치', zorder=5)
                    plt.legend()
                
                plt.colorbar(h[3], ax=ax, label='시선 빈도')
                plt.xlabel('X 좌표 (픽셀)')
                plt.ylabel('Y 좌표 (픽셀)')
                plt.title(f'단계 {actual_step_number} 시선 히트맵: {step_info["target_element"] if step_number < len(self.scenario_steps) else ""}')
                plt.grid(True, alpha=0.3)
                
                # 저장
                heatmap_file = os.path.join(self.data_dir, f"step_{actual_step_number}_heatmap.png")
                plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. 시선 궤적 생성
                if len(step_gaze_data) > 1:
                    plt.figure(figsize=(12, 8))
                    plt.plot(x_coords, y_coords, 'b-', alpha=0.6, linewidth=1)
                    plt.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', label='시작', zorder=5)
                    plt.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='o', label='끝', zorder=5)
                    
                    if step_click_events:
                        click_x = [event['x'] for event in step_click_events]
                        click_y = [event['y'] for event in step_click_events]
                        plt.scatter(click_x, click_y, c='red', s=150, marker='*', label='클릭', zorder=5)
                    
                    plt.xlabel('X 좌표 (픽셀)')
                    plt.ylabel('Y 좌표 (픽셀)')
                    plt.title(f'단계 {actual_step_number} 시선 궤적: {step_info["target_element"] if step_number < len(self.scenario_steps) else ""}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # 저장
                    trajectory_file = os.path.join(self.data_dir, f"step_{actual_step_number}_trajectory.png")
                    plt.savefig(trajectory_file, dpi=300, bbox_inches='tight')
                    plt.close()
            else:
                # 시선 데이터가 없는 경우 빈 히트맵 생성
                plt.figure(figsize=(12, 8))
                plt.text(0.5, 0.5, f'단계 {actual_step_number}\n시선 데이터 없음', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
                plt.title(f'단계 {actual_step_number}: {step_info["target_element"] if step_number < len(self.scenario_steps) else ""}')
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                
                # 저장
                heatmap_file = os.path.join(self.data_dir, f"step_{actual_step_number}_heatmap.png")
                plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. 단계별 데이터 JSON 저장
            step_data = {
                'step_number': actual_step_number,  # 실제 step 번호 (1, 2, 3, ...)
                'array_index': step_number,  # 배열 인덱스
                'step_info': step_info if step_number < len(self.scenario_steps) else None,
                'gaze_data': step_gaze_data,
                'click_events': step_click_events,
                'gaze_count': len(step_gaze_data),
                'click_count': len(step_click_events)
            }
            
            step_json_file = os.path.join(self.data_dir, f"step_{actual_step_number}_data.json")
            with open(step_json_file, 'w', encoding='utf-8') as f:
                json.dump(step_data, f, ensure_ascii=False, indent=2)
            
            # 생성 완료 표시
            self.step_visualizations_created.add(step_number)
            
            self.logger.info(f"단계 {actual_step_number} 시각화 생성 완료")
            print(f"📊 단계 {actual_step_number} 시각화 생성 완료")
            
        except Exception as e:
            self.log_error("STEP_VISUALIZATION", f"단계 {step_number} 시각화 생성 실패: {e}", traceback.format_exc())

    def create_step_summary_report(self):
        """단계별 요약 리포트 생성"""
        try:
            report_data = {
                'scenario_name': self.current_scenario,
                'timestamp': self.timestamp,
                'total_steps': len(self.scenario_steps),
                'completed_steps': len(self.step_visualizations_created),
                'steps': []
            }
            
            for step_num in range(len(self.scenario_steps)):
                step_info = self.scenario_steps[step_num]
                actual_step_number = step_info['step'] + 1  # 실제 step 번호 (1, 2, 3, ...)
                step_gaze_count = len(self.step_gaze_data.get(step_num, []))
                step_click_count = len(self.step_click_events.get(step_num, []))
                has_visualization = step_num in self.step_visualizations_created
                
                step_summary = {
                    'step_number': actual_step_number,  # 실제 step 번호 (1, 2, 3, ...)
                    'array_index': step_num,  # 배열 인덱스
                    'target_element': step_info['target_element'],
                    'category': step_info['category'],
                    'difficulty': step_info['difficulty'],
                    'gaze_data_count': step_gaze_count,
                    'click_events_count': step_click_count,
                    'visualization_created': has_visualization,
                    'files': {
                        'heatmap': f"step_{actual_step_number}_heatmap.png" if has_visualization else None,
                        'trajectory': f"step_{actual_step_number}_trajectory.png" if has_visualization else None,
                        'data': f"step_{actual_step_number}_data.json" if has_visualization else None
                    }
                }
                report_data['steps'].append(step_summary)
            
            # 리포트 저장
            report_file = os.path.join(self.data_dir, "step_summary_report.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            # 텍스트 리포트도 생성
            text_report_file = os.path.join(self.data_dir, "step_summary_report.txt")
            with open(text_report_file, 'w', encoding='utf-8') as f:
                f.write(f"헬 난이도 키오스크 시나리오 요약 리포트\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"시나리오: {self.current_scenario}\n")
                f.write(f"수집 시간: {self.timestamp}\n")
                f.write(f"총 단계: {len(self.scenario_steps)}\n")
                f.write(f"완료된 단계: {len(self.step_visualizations_created)}\n\n")
                
                f.write("단계별 상세 정보:\n")
                f.write("-" * 30 + "\n")
                
                for step_summary in report_data['steps']:
                    f.write(f"단계 {step_summary['step_number']}: {step_summary['target_element']}\n")
                    f.write(f"  카테고리: {step_summary['category']}\n")
                    f.write(f"  난이도: {step_summary['difficulty']}\n")
                    f.write(f"  시선 데이터: {step_summary['gaze_data_count']}개\n")
                    f.write(f"  클릭 이벤트: {step_summary['click_events_count']}개\n")
                    f.write(f"  시각화 생성: {'✅' if step_summary['visualization_created'] else '❌'}\n")
                    f.write("\n")
            
            self.logger.info(f"단계별 요약 리포트 생성 완료: {report_file}")
            print(f"📋 단계별 요약 리포트 생성 완료")
            
        except Exception as e:
            self.log_error("SUMMARY_REPORT", f"요약 리포트 생성 실패: {e}", traceback.format_exc())

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
            self.log_error("SCREENSHOT_START", f"스크린샷 캡처 시작 실패: {e}", traceback.format_exc())
    
    def stop_screenshot_capture(self):
        """스크린샷 캡처 중지"""
        try:
            self.screenshot_active = False
            if self.screenshot_thread:
                self.screenshot_thread.join(timeout=1.0)
            self.logger.info("스크린샷 캡처 중지")
        except Exception as e:
            self.log_error("SCREENSHOT_STOP", f"스크린샷 캡처 중지 실패: {e}", traceback.format_exc())
    
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
                self.log_error("SCREENSHOT_CAPTURE", f"스크린샷 캡처 실패: {e}", traceback.format_exc())
                time.sleep(1.0)
    
    def create_enhanced_heatmap(self, step_number: int):
        """향상된 히트맵 생성 (스크린샷 + 시선 데이터)"""
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
            
            # 해당 단계의 스크린샷 찾기
            step_screenshots = [s for s in self.screenshot_data if s['step'] == step_number]
            
            if not step_screenshots:
                self.logger.warning(f"단계 {actual_step_number}의 스크린샷이 없습니다.")
                return
            
            # 가장 최근 스크린샷 사용
            latest_screenshot = step_screenshots[-1]
            
            # Base64 이미지 디코딩
            img_data = base64.b64decode(latest_screenshot['image_base64'])
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img)
            
            # matplotlib 설정
            plt.rcParams['font.family'] = 'Malgun Gothic'
            plt.rcParams['axes.unicode_minus'] = False
            
            # 히트맵 생성
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # 스크린샷을 배경으로 표시
            ax.imshow(img_array)
            
            # 시선 데이터가 있으면 히트맵 오버레이
            if step_gaze_data:
                x_coords = [data['x'] for data in step_gaze_data]
                y_coords = [data['y'] for data in step_gaze_data]
                
                # 히트맵 생성 (반투명)
                h = ax.hist2d(x_coords, y_coords, bins=30, cmap='hot', alpha=0.6)
                
                # 클릭 위치 표시
                if step_click_events:
                    click_x = [event['x'] for event in step_click_events]
                    click_y = [event['y'] for event in step_click_events]
                    ax.scatter(click_x, click_y, c='red', s=200, marker='*', label='클릭 위치', zorder=5)
                    ax.legend()
                
                # 컬러바 추가
                plt.colorbar(h[3], ax=ax, label='시선 빈도')
            
            # 제목과 레이블
            ax.set_title(f'단계 {actual_step_number} 시선 히트맵: {step_info["target_element"] if step_number < len(self.scenario_steps) else ""}\n(스크린샷 + 시선 데이터)', fontsize=14)
            ax.set_xlabel('X 좌표 (픽셀)')
            ax.set_ylabel('Y 좌표 (픽셀)')
            
            # 저장
            heatmap_file = os.path.join(self.data_dir, f"step_{actual_step_number}_enhanced_heatmap.png")
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 스크린샷도 별도 저장
            screenshot_file = os.path.join(self.data_dir, f"step_{actual_step_number}_screenshot.png")
            img.save(screenshot_file)
            
            self.logger.info(f"단계 {actual_step_number} 향상된 히트맵 생성 완료")
            print(f"📊 단계 {actual_step_number} 향상된 히트맵 생성 완료")
            
        except Exception as e:
            self.log_error("ENHANCED_HEATMAP", f"향상된 히트맵 생성 실패: {e}", traceback.format_exc())

def main():
    """헬 난이도 키오스크 시나리오 수집 메인 함수"""
    print("🔥 헬 난이도 키오스크 시나리오 데이터 수집기")
    print("고령자 UI 사용성 연구를 위한 복잡한 시나리오 데이터 수집")
    print("="*60)
    
    try:
        # 수집기 생성
        collector = KioskScenarioCollector("R1_kiosk_hell")
        
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