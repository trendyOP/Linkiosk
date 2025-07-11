import sys
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QFileDialog, QLineEdit, QTextEdit, QMessageBox, QScrollArea,
    QCheckBox
)
from PyQt5.QtCore import Qt
import subprocess
import threading
import os
from PyQt5.QtCore import QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtCore import QTimer
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import pyqtSlot, QObject
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
from PIL import ImageGrab, Image
import io
import base64

# 시선추적 SDK import
try:
    from eyeware import beam_eye_tracker as bet
    from screeninfo import get_monitors
    EYEWARE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Eyeware SDK not available. Eye tracking will be disabled. Error: {e}")
    EYEWARE_AVAILABLE = False

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ScenarioStepWidget(QWidget):
    def __init__(self, step_num, text=""):
        super().__init__()
        self.layout = QHBoxLayout()
        self.label = QLabel(f"단계 {step_num}")
        self.edit = QLineEdit(text)
        self.remove_btn = QPushButton("-")
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.edit)
        self.layout.addWidget(self.remove_btn)
        self.setLayout(self.layout)

class CustomWebEnginePage(QWebEnginePage):
    def __init__(self, parent=None, logger=None):
        super().__init__(parent)
        self.logger = logger

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        if self.logger:
            self.logger(f"[WebView JS] {message} (line {lineNumber})")
        else:
            print(f"[WebView JS] {message} (line {lineNumber})")

class Bridge(QObject):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    @pyqtSlot()
    def clicked(self):
        self.parent.advance_scenario_step()

    @pyqtSlot(str)
    def clickedWithInfo(self, element_info_json):
        import json
        try:
            element_info = json.loads(element_info_json)
            if self.parent.check_scenario_step_completion(element_info):
                self.parent.advance_scenario_step()
        except Exception as e:
            print(f"클릭 정보 처리 오류: {e}")

    @pyqtSlot(str)
    def handleWebMessage(self, message_json):
        """웹뷰에서 온 메시지를 처리"""
        self.parent.handle_web_message(message_json)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UIUX 인증 프로그램")
        self.setGeometry(100, 100, 1400, 900)
        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.main_layout = QHBoxLayout()
        self.central.setLayout(self.main_layout)

        # 기본값 설정
        self.default_ui_path = r"C:\Linkiosk-main\kiosk web_v2\kiosk web_v2"
        self.default_scenario_steps = [
            "아메리카노",
            "결제"
        ]

        # 시나리오 상태 관리
        self.current_step = 0
        self.scenario_completed = False
        
        # 시나리오 상태 머신 추가
        self.scenario_states = {
            'current_step': 0,
            'completed_steps': set(),  # 완료된 단계들
            'failed_steps': set(),     # 실패한 단계들
            'skipped_steps': set(),    # 건너뛴 단계들
            'step_history': [],        # 단계별 행동 이력
            'last_action': None,       # 마지막 행동
            'last_action_time': None,  # 마지막 행동 시간
            'retry_count': {},         # 단계별 재시도 횟수
            'max_retries': 3          # 최대 재시도 횟수
        }
        
        # 장바구니 상태 추적
        self.cart_state = {
            'itemCount': 0,
            'totalItems': 0,
            'totalPrice': 0,
            'items': [],
            'step': 0
        }
        self.cart_history = []  # 장바구니 변경 이력

        # 시선추적 관련 변수들
        self.eye_tracker = None
        self.tracking_active = False
        self.gaze_data = []
        self.screen_width, self.screen_height = self.get_screen_resolution()
        self.gaze_thread = None
        self.last_update_timestamp = None
        
        # 시나리오 데이터 저장 관련
        self.scenario_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scenario_data_dir = f"kiosk_data/uiux_certification_{self.scenario_timestamp}"
        os.makedirs(self.scenario_data_dir, exist_ok=True)
        
        # 스크린샷 및 히트맵 관련
        self.current_screenshot_path = None
        self.click_events = []
        self.current_gaze_segment = []
        self.scenario_started = False

        # 시나리오 영역
        self.scenario_layout = QVBoxLayout()
        self.scenario_label = QLabel("시나리오 단계")
        self.scenario_layout.addWidget(self.scenario_label)
        self.scenario_steps = []
        self.scenario_step_widgets = []
        self.steps_area = QVBoxLayout()
        self.steps_container = QWidget()
        self.steps_container.setLayout(self.steps_area)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.steps_container)
        self.scenario_layout.addWidget(self.scroll)
        self.add_step_btn = QPushButton("+ 단계 추가")
        self.add_step_btn.clicked.connect(self.add_scenario_step)
        self.scenario_layout.addWidget(self.add_step_btn)
        self.main_layout.addLayout(self.scenario_layout, 2)

        # 중앙: 웹뷰 (웹앱 표시)
        self.webview = QWebEngineView()
        self.webview.setPage(CustomWebEnginePage(self.webview, logger=self.log))
        self.webview.setHtml("<h2>여기에 UI 웹앱이 표시됩니다.</h2>")
        
        # 웹뷰 설정 개선 - React 앱 호환성을 위한 완전한 설정
        settings = self.webview.settings()
        
        # 핵심 설정 - React 앱 로드를 위해 필수
        settings.setAttribute(settings.JavascriptEnabled, True)
        settings.setAttribute(settings.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(settings.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(settings.AllowRunningInsecureContent, True)
        settings.setAttribute(settings.AllowWindowActivationFromJavaScript, True)
        
        # localhost 호환성을 위한 추가 설정
        try:
            settings.setAttribute(settings.LocalContentCanAccessRemoteUrls, True)
            settings.setAttribute(settings.LocalContentCanAccessFileUrls, True)
            settings.setAttribute(settings.AllowRunningInsecureContent, True)
            settings.setAttribute(settings.AllowWindowActivationFromJavaScript, True)
        except AttributeError:
            pass
        
        # 추가 설정 (PyQt5 버전에 따라 지원되지 않을 수 있음)
        additional_settings = [
            ('JavascriptCanAccessClipboard', True),
            ('JavascriptCanOpenWindows', True),
            ('JavascriptCanPaste', True),
            ('LocalStorageEnabled', True),
            ('PluginsEnabled', True),
            ('ScreenCaptureEnabled', True),
            ('WebGLEnabled', True),
            ('DnsPrefetchEnabled', True),
            ('FocusOnNavigationEnabled', True),
            ('FullScreenSupportEnabled', True),
            ('HyperlinkAuditingEnabled', True),
            ('LinksIncludedInFocusChain', True),
            ('SpatialNavigationEnabled', True),
        ]
        
        for setting_name, value in additional_settings:
            try:
                setting_attr = getattr(settings, setting_name)
                settings.setAttribute(setting_attr, value)
            except AttributeError:
                print(f"{setting_name} 설정을 건너뜁니다 (PyQt5 버전 제한)")
        
        # 보안 관련 설정 (React 개발용)
        try:
            settings.setAttribute(settings.XSSAuditorEnabled, False)
        except AttributeError:
            print("XSSAuditorEnabled 설정을 건너뜁니다 (PyQt5 버전 제한)")
        
        try:
            settings.setAttribute(settings.WebSecurityEnabled, False)
        except AttributeError:
            print("WebSecurityEnabled 설정을 건너뜁니다 (PyQt5 버전 제한)")
        
        self.main_layout.addWidget(self.webview, 4)
        self.webview.loadFinished.connect(self.on_webview_load_finished)

        # UI/시나리오 불러오기 및 실행
        self.control_layout = QVBoxLayout()
        self.ui_label = QLabel("UI 프로젝트 경로:")
        self.ui_path_edit = QLineEdit()
        self.ui_browse_btn = QPushButton("UI 불러오기")
        self.ui_browse_btn.clicked.connect(self.browse_ui)
        self.port_label = QLabel("포트:")
        self.port_edit = QLineEdit("3000")
        self.port_edit.setMaximumWidth(80)
        self.static_build_checkbox = QCheckBox("정적 빌드로 실행 (권장)")
        self.static_build_checkbox.setChecked(True)
        self.scenario_file_btn = QPushButton("시나리오 불러오기")
        self.scenario_file_btn.clicked.connect(self.browse_scenario)
        self.scenario_save_btn = QPushButton("시나리오 저장")
        self.scenario_save_btn.clicked.connect(self.save_scenario)
        self.run_btn = QPushButton("실행")
        self.run_btn.clicked.connect(self.run_certification)
        
        # 시나리오 제어 버튼들
        self.reset_btn = QPushButton("시나리오 초기화")
        self.reset_btn.clicked.connect(self.reset_scenario)
        
        # 테스트용 버튼 추가
        self.test_btn = QPushButton("네이버 테스트")
        self.test_btn.clicked.connect(self.test_external_site)
        self.control_layout.addWidget(self.ui_label)
        self.control_layout.addWidget(self.ui_path_edit)
        self.control_layout.addWidget(self.ui_browse_btn)
        port_hbox = QHBoxLayout()
        port_hbox.addWidget(self.port_label)
        port_hbox.addWidget(self.port_edit)
        port_hbox.addStretch(1)
        self.control_layout.addLayout(port_hbox)
        self.control_layout.addWidget(self.static_build_checkbox)
        self.control_layout.addWidget(self.scenario_file_btn)
        self.control_layout.addWidget(self.scenario_save_btn)
        self.control_layout.addWidget(self.run_btn)
        self.control_layout.addWidget(self.reset_btn)
        self.control_layout.addWidget(self.test_btn)
        self.main_layout.addLayout(self.control_layout, 1)

        # 진행상황/로그 영역(우측)
        self.status_layout = QVBoxLayout()
        self.status_label = QLabel("진행상황/로그")
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        
        # 시선추적 상태 표시
        self.eye_tracking_status = QLabel("👁️ 시선추적: 비활성")
        self.eye_tracking_status.setStyleSheet("color: gray; font-weight: bold;")
        
        # 데이터 수집 상태 표시
        self.data_collection_status = QLabel("📊 데이터 수집: 대기 중")
        self.data_collection_status.setStyleSheet("color: gray; font-weight: bold;")
        
        self.status_layout.addWidget(self.status_label)
        self.status_layout.addWidget(self.eye_tracking_status)
        self.status_layout.addWidget(self.data_collection_status)
        self.status_layout.addWidget(self.status_text)
        self.main_layout.addLayout(self.status_layout, 2)

        # UI 프로젝트 경로 설정 (기본값)
        self.ui_path_edit.setText(self.default_ui_path)

        # 기본 시나리오 단계들 추가
        for step_text in self.default_scenario_steps:
            self.add_scenario_step(step_text)

        # QWebChannel 브릿지
        self.channel = QWebChannel()
        self.bridge = Bridge(self)
        self.channel.registerObject('pyBridge', self.bridge)
        self.webview.page().setWebChannel(self.channel)

    def add_scenario_step(self, text="", *_):
        if isinstance(text, bool):
            text = ""
        step_num = len(self.scenario_step_widgets) + 1
        step_widget = ScenarioStepWidget(step_num, text)
        step_widget.remove_btn.clicked.connect(lambda: self.remove_scenario_step(step_widget))
        self.steps_area.addWidget(step_widget)
        self.scenario_step_widgets.append(step_widget)

    def remove_scenario_step(self, widget):
        self.steps_area.removeWidget(widget)
        widget.setParent(None)
        self.scenario_step_widgets.remove(widget)
        # 단계 번호 재정렬
        for i, w in enumerate(self.scenario_step_widgets):
            w.label.setText(f"단계 {i+1}")

    def browse_ui(self):
        path = QFileDialog.getExistingDirectory(self, "UI 프로젝트 폴더 선택")
        if path:
            self.ui_path_edit.setText(path)

    def browse_scenario(self):
        path, _ = QFileDialog.getOpenFileName(self, "시나리오 파일 불러오기", filter="JSON Files (*.json)")
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.clear_scenario_steps()
                for step in data.get('steps', []):
                    self.add_scenario_step(step)
                self.log(f"시나리오 불러오기 완료: {path}")
            except Exception as e:
                QMessageBox.warning(self, "오류", f"시나리오 파일을 불러올 수 없습니다: {e}")

    def clear_scenario_steps(self):
        for w in self.scenario_step_widgets:
            self.steps_area.removeWidget(w)
            w.setParent(None)
        self.scenario_step_widgets = []

    def run_certification(self):
        # 시나리오 시작 시 시선추적 및 초기 스크린샷 시작
        if not self.scenario_started:
            self.log("🚀 시나리오 시작 - 시선추적 및 데이터 수집 시작")
            self.scenario_started = True
            
            # 시선추적 시작
            if self.start_eye_tracker():
                # 초기 스크린샷 촬영
                self.take_screenshot("screenshot_initial.png")
                self.log("📸 초기 스크린샷 촬영 완료")
            else:
                self.log("⚠️ 시선추적을 시작할 수 없습니다. 데이터 수집 없이 진행합니다.")
        
        # 내부 서버 실행 없이, 입력 포트로 바로 접속
        port = 3000  # 고정
        url = QUrl(f"http://localhost:{port}")
        self.log(f"웹뷰에 바로 접속: {url.toString()}")
        self.webview.setUrl(url)
        # JS 인젝션: 클릭 이벤트를 PyQt5로 전달
        QTimer.singleShot(2000, self.inject_js_bridge)

    def launch_ui_project(self, ui_path, port, use_static_build):
        try:
            # React 기준: npm start 또는 빌드+serve
            if os.path.exists(os.path.join(ui_path, 'package.json')):
                self.log("package.json 발견: npm/yarn 프로젝트로 인식")
                npm_cmd = 'npm.cmd' if os.name == 'nt' else 'npm'
                yarn_cmd = 'yarn.cmd' if os.name == 'nt' else 'yarn'
                if use_static_build:
                    try:
                        self.log("npm install 실행 중...")
                        subprocess.run([npm_cmd, 'install'], cwd=ui_path, check=True, capture_output=True, encoding='utf-8', errors='ignore')
                        self.log("npm run build 실행 중...")
                        subprocess.run([npm_cmd, 'run', 'build'], cwd=ui_path, check=True, capture_output=True, encoding='utf-8', errors='ignore')
                        self.log("Python 내장 HTTP 서버로 React 앱 서빙...")
                        # PyQt5 웹뷰 호환성을 위해 Python 내장 HTTP 서버 사용
                        self._start_python_http_server(ui_path, port)
                        return
                        self.log("UI 빌드 서버가 실행되었습니다. (정적 빌드)")
                        threading.Thread(target=self._read_ui_log, daemon=True).start()
                        # 서버 시작 후 더 오래 기다린 후 웹뷰 로드 (React 앱이 완전히 준비될 때까지)
                        QTimer.singleShot(20000, lambda: self.load_webview(port))
                    except Exception as e2:
                        self.log(f"UI 실행 실패: {e2}")
                        QMessageBox.critical(self, "UI 실행 오류", f"UI 실행에 실패했습니다: {e2}")
                else:
                    try:
                        self.log("npm install 실행 중...")
                        subprocess.run([npm_cmd, 'install'], cwd=ui_path, check=True, capture_output=True, encoding='utf-8', errors='ignore')
                        self.log("npm start 실행 중...")
                        env = os.environ.copy()
                        # 환경변수로 포트 지정 (Windows/Mac/Linux 모두 지원)
                        if os.name == 'nt':
                            env['PORT'] = str(port)
                        else:
                            env['PORT'] = str(port)
                        self.ui_proc = subprocess.Popen([npm_cmd, 'start'], cwd=ui_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, encoding='utf-8', errors='ignore')
                        self.log("UI 개발 서버가 실행되었습니다. (포트는 프로젝트 설정에 따름)")
                        threading.Thread(target=self._read_ui_log, daemon=True).start()
                        QTimer.singleShot(2000, lambda: self.load_webview(port))
                    except Exception as e:
                        self.log(f"UI 실행 실패: {e}")
                        QMessageBox.critical(self, "UI 실행 오류", f"UI 실행에 실패했습니다: {e}")
            else:
                self.log("지원하지 않는 UI 프로젝트 구조입니다.")
                QMessageBox.warning(self, "UI 실행 오류", "지원하지 않는 UI 프로젝트 구조입니다.")
        except Exception as e:
            self.log(f"UI 실행 중 예외 발생: {e}")
            QMessageBox.critical(self, "UI 실행 오류", f"UI 실행 중 예외 발생: {e}")

    def _read_ui_log(self):
        if not hasattr(self, 'ui_proc') or not self.ui_proc:
            return
        try:
            for line in self.ui_proc.stdout:
                try:
                    line = line.rstrip()
                    if line:
                        self.log(line)
                except UnicodeDecodeError:
                    # 인코딩 오류 시 무시하고 계속 진행
                    continue
        except Exception as e:
            self.log(f"로그 읽기 오류: {e}")

    def save_scenario(self):
        steps = [w.edit.text().strip() for w in self.scenario_step_widgets if w.edit.text().strip()]
        if not steps:
            QMessageBox.warning(self, "저장 오류", "저장할 시나리오 단계가 없습니다.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "시나리오 파일 저장", filter="JSON Files (*.json)")
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump({"steps": steps}, f, ensure_ascii=False, indent=2)
                self.log(f"시나리오 저장 완료: {path}")
            except Exception as e:
                QMessageBox.warning(self, "오류", f"시나리오 파일을 저장할 수 없습니다: {e}")

    def log(self, msg):
        self.status_text.append(msg)

    def find_available_port(self, start_port):
        import socket
        for port in range(start_port, start_port+100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("localhost", port))
                    s.listen(1)
                    return port
                except OSError:
                    continue
        self.log("사용 가능한 포트를 찾지 못했습니다. 기본값 3000 사용")
        return 3000

    def _start_python_http_server(self, ui_path, port):
        """Python 내장 HTTP 서버로 정적 파일 서빙"""
        try:
            from http.server import HTTPServer, SimpleHTTPRequestHandler
            import threading
            
            build_path = os.path.join(ui_path, 'build')
            if not os.path.exists(build_path):
                self.log(f"❌ build 폴더를 찾을 수 없습니다: {build_path}")
                return
            
            class CustomHandler(SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=build_path, **kwargs)
                
                def end_headers(self):
                    # CORS 헤더 추가
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    super().end_headers()
                
                def do_GET(self):
                    # React 앱의 경우 모든 경로를 index.html로 리다이렉트
                    if self.path == '/' or self.path == '/index.html':
                        self.path = '/index.html'
                    elif not os.path.exists(os.path.join(build_path, self.path.lstrip('/'))):
                        # 파일이 없으면 index.html로 리다이렉트 (React Router 지원)
                        self.path = '/index.html'
                    super().do_GET()
                
                def log_message(self, format, *args):
                    # 로그 비활성화
                    pass
            
            def run_server():
                server = HTTPServer(('127.0.0.1', port), CustomHandler)
                self.log(f"✅ Python HTTP 서버 시작: http://127.0.0.1:{port}")
                server.serve_forever()
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # 서버 시작 후 웹뷰 로드
            QTimer.singleShot(2000, lambda: self.load_webview(port))
            
        except Exception as e:
            self.log(f"❌ Python HTTP 서버 시작 실패: {e}")

    def load_webview(self, port):
        url = QUrl(f"http://localhost:{port}")
        self.log(f"웹뷰에 바로 접속: {url.toString()}")
        self.webview.setUrl(url)
        # JS 인젝션: 클릭 이벤트를 PyQt5로 전달
        QTimer.singleShot(2000, self.inject_js_bridge)

    def inject_js_bridge(self):
        js = '''
        try {
            // QWebChannel 라이브러리를 수동으로 로드
            if (!window.QWebChannel) {
                console.log('QWebChannel 라이브러리 수동 로드 시도...');
                const script = document.createElement('script');
                script.src = 'qrc:///qtwebchannel/qwebchannel.js';
                script.onload = function() {
                    console.log('QWebChannel 라이브러리 로드 성공!');
                    initializeBridge();
                };
                script.onerror = function() {
                    console.log('QWebChannel 라이브러리 로드 실패 - 대체 방법 사용');
                    setupDirectClickHandler();
                };
                document.head.appendChild(script);
            } else {
                console.log('QWebChannel 라이브러리 이미 존재');
                initializeBridge();
            }
            
            function initializeBridge() {
                if (window.qt && window.qt.webChannelTransport) {
                    console.log('QWebChannel 초기화 시작...');
                    new QWebChannel(qt.webChannelTransport, function(channel) {
                        console.log('QWebChannel 연결 성공!');
                        window.pyBridge = channel.objects.pyBridge;
                        setupClickHandler();
                        setupMessageHandler();
                    });
                } else {
                    console.log('QWebChannel transport 없음 - 대체 방법 사용');
                    setupDirectClickHandler();
                }
            }
            
            function setupClickHandler() {
                // 기존 클릭 이벤트 리스너 제거
                if (window._pyqtClickHandler) {
                    document.removeEventListener('click', window._pyqtClickHandler, true);
                }
                
                // 새로운 클릭 이벤트 리스너 추가
                window._pyqtClickHandler = function(event) {
                    try {
                        const element = event.target;
                        const elementInfo = {
                            tagName: element.tagName,
                            className: element.className,
                            textContent: element.textContent || element.innerText || '',
                            id: element.id,
                            dataName: element.getAttribute('data-name') || ''
                        };
                        
                        console.log('클릭 감지됨:', elementInfo);
                        
                        if (window.pyBridge && window.pyBridge.clicked) {
                            // 클릭된 요소 정보를 PyQt로 전달
                            window.pyBridge.clickedWithInfo(JSON.stringify(elementInfo));
                        } else {
                            console.log('PyQt bridge 또는 clicked 메서드 없음');
                        }
                    } catch (e) {
                        console.log('클릭 처리 오류:', e);
                    }
                };
                
                document.addEventListener('click', window._pyqtClickHandler, true);
                console.log('PyQt bridge 초기화 완료 - 클릭 이벤트 리스너 등록됨');
            }
            
            function setupMessageHandler() {
                // React 앱의 pyqtBridge 객체를 PyQt5와 연결
                if (window.pyqtBridge) {
                    console.log('React 앱의 pyqtBridge 객체 발견 - 메시지 핸들러 설정');
                    
                    // 기존 메서드들을 PyQt5로 전달하도록 오버라이드
                    const originalNotifyCartChange = window.pyqtBridge.notifyCartChange;
                    const originalNotifyClick = window.pyqtBridge.notifyClick;
                    
                    window.pyqtBridge.notifyCartChange = function(cartState) {
                        console.log('장바구니 상태 변경 감지:', cartState);
                        
                        // 원본 메서드 호출
                        if (originalNotifyCartChange) {
                            originalNotifyCartChange(cartState);
                        }
                        
                        // PyQt5로 전달
                        if (window.pyBridge && window.pyBridge.handleWebMessage) {
                            window.pyBridge.handleWebMessage(JSON.stringify({
                                type: 'cartChange',
                                data: cartState
                            }));
                        }
                    };
                    
                    window.pyqtBridge.notifyClick = function(elementInfo) {
                        console.log('클릭 이벤트 감지:', elementInfo);
                        
                        // 원본 메서드 호출
                        if (originalNotifyClick) {
                            originalNotifyClick(elementInfo);
                        }
                        
                        // PyQt5로 전달
                        if (window.pyBridge && window.pyBridge.handleWebMessage) {
                            window.pyBridge.handleWebMessage(JSON.stringify({
                                type: 'click',
                                data: elementInfo
                            }));
                        }
                    };
                    
                    console.log('메시지 핸들러 설정 완료');
                } else {
                    console.log('React 앱의 pyqtBridge 객체가 아직 로드되지 않음');
                    // React 앱이 로드될 때까지 대기
                    setTimeout(setupMessageHandler, 1000);
                }
            }
            
            function setupDirectClickHandler() {
                console.log('직접 클릭 핸들러 설정');
                if (window._directClickHandler) {
                    document.removeEventListener('click', window._directClickHandler, true);
                }
                
                window._directClickHandler = function(event) {
                    console.log('직접 클릭 감지됨:', event.target.tagName, event.target.className);
                    console.log('클릭 좌표:', event.clientX, event.clientY);
                };
                
                document.addEventListener('click', window._directClickHandler, true);
                console.log('직접 클릭 핸들러 등록 완료');
            }
            
        } catch (e) {
            console.log('JS 브릿지 초기화 오류:', e);
        }
        '''
        self.webview.page().runJavaScript(js)

    def advance_scenario_step(self):
        """시나리오 단계를 진행합니다."""
        if self.current_step < len(self.scenario_step_widgets):
            # 현재 단계를 완료로 표시
            self.complete_current_step()
            
            # 다음 단계로 진행
            self.current_step += 1
            self.scenario_states['current_step'] = self.current_step
            
            self.update_scenario_ui()
            self.log(f"✅ 단계 {self.current_step}로 자동 진행")
            
            # 시나리오 완료 확인
            if self.current_step > len(self.scenario_step_widgets):
                self.complete_scenario()
        else:
            self.complete_scenario()

    def complete_current_step(self):
        """현재 단계를 완료로 표시합니다."""
        if self.current_step < len(self.scenario_step_widgets):
            self.scenario_states['completed_steps'].add(self.current_step)
            self.log(f"🎯 단계 {self.current_step} 완료: {self.scenario_step_widgets[self.current_step].edit.text()}")

    def fail_current_step(self, reason="알 수 없는 이유"):
        """현재 단계를 실패로 표시합니다."""
        if self.current_step < len(self.scenario_step_widgets):
            self.scenario_states['failed_steps'].add(self.current_step)
            retry_count = self.scenario_states['retry_count'].get(self.current_step, 0) + 1
            self.scenario_states['retry_count'][self.current_step] = retry_count
            
            self.log(f"❌ 단계 {self.current_step} 실패: {reason} (재시도 {retry_count}/{self.scenario_states['max_retries']})")
            
            # 최대 재시도 횟수 초과 시 다음 단계로 강제 진행
            if retry_count >= self.scenario_states['max_retries']:
                self.log(f"⚠️ 단계 {self.current_step} 최대 재시도 횟수 초과 - 다음 단계로 강제 진행")
                self.advance_scenario_step()

    def skip_current_step(self, reason="사용자가 건너뜀"):
        """현재 단계를 건너뜁니다."""
        if self.current_step < len(self.scenario_step_widgets):
            self.scenario_states['skipped_steps'].add(self.current_step)
            self.log(f"⏭️ 단계 {self.current_step} 건너뜀: {reason}")
            self.advance_scenario_step()

    def complete_scenario(self):
        """시나리오를 완료합니다."""
        self.scenario_completed = True
        self.log("🎉 시나리오 완료!")
        
        # 시나리오 데이터 저장
        if self.scenario_started:
            self.save_scenario_data()
        
        # 시선추적 중지
        self.stop_eye_tracker()
        
        # 완료 통계 출력
        total_steps = len(self.scenario_step_widgets)
        completed = len(self.scenario_states['completed_steps'])
        failed = len(self.scenario_states['failed_steps'])
        skipped = len(self.scenario_states['skipped_steps'])
        
        self.log(f"📊 시나리오 결과:")
        self.log(f"  - 총 단계: {total_steps}")
        self.log(f"  - 완료: {completed}")
        self.log(f"  - 실패: {failed}")
        self.log(f"  - 건너뜀: {skipped}")
        self.log(f"  - 성공률: {completed/total_steps*100:.1f}%")
        
        # 시선추적 데이터 통계
        if self.scenario_started:
            self.log(f"👁️ 시선추적 데이터:")
            self.log(f"  - 총 시선 데이터 포인트: {len(self.gaze_data)}")
            self.log(f"  - 총 클릭 이벤트: {len(self.click_events)}")
            self.log(f"  - 데이터 저장 위치: {self.scenario_data_dir}")

    def reset_scenario(self):
        """시나리오를 초기화합니다."""
        self.current_step = 0
        self.scenario_completed = False
        self.scenario_states = {
            'current_step': 0,
            'completed_steps': set(),
            'failed_steps': set(),
            'skipped_steps': set(),
            'step_history': [],
            'last_action': None,
            'last_action_time': None,
            'retry_count': {},
            'max_retries': 3
        }
        self.cart_state = {
            'itemCount': 0,
            'totalItems': 0,
            'totalPrice': 0,
            'items': [],
            'step': 0
        }
        self.cart_history = []
        
        # 시선추적 관련 변수 초기화
        self.scenario_started = False
        self.gaze_data = []
        self.click_events = []
        self.current_gaze_segment = []
        self.current_screenshot_path = None
        
        # 새로운 시나리오 데이터 디렉토리 생성
        self.scenario_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.scenario_data_dir = f"kiosk_data/uiux_certification_{self.scenario_timestamp}"
        os.makedirs(self.scenario_data_dir, exist_ok=True)
        
        self.update_scenario_ui()
        self.log("🔄 시나리오 초기화 완료")
        self.log(f"📁 새로운 데이터 디렉토리: {self.scenario_data_dir}")
        
        # 상태 업데이트
        self.update_eye_tracking_status(False)
        self.update_data_collection_status(False)

    def record_action(self, action_type, details=None):
        """사용자 행동을 기록합니다."""
        action = {
            'type': action_type,
            'details': details,
            'timestamp': time.time(),
            'step': self.current_step,
            'cart_state': self.cart_state.copy()
        }
        
        self.scenario_states['step_history'].append(action)
        self.scenario_states['last_action'] = action
        self.scenario_states['last_action_time'] = time.time()
        
        self.log(f"📝 행동 기록: {action_type} (단계 {self.current_step})")

    def update_scenario_ui(self):
        """시나리오 UI를 업데이트하여 각 단계의 상태를 시각적으로 표시합니다."""
        for i, w in enumerate(self.scenario_step_widgets):
            step_num = i + 1
            status_text = f"단계 {step_num}"
            
            # 단계별 상태에 따른 스타일 적용
            if i in self.scenario_states['completed_steps']:
                # 완료된 단계
                w.label.setStyleSheet('font-weight: bold; color: green;')
                status_text += " ✅"
            elif i in self.scenario_states['failed_steps']:
                # 실패한 단계
                w.label.setStyleSheet('font-weight: bold; color: red;')
                retry_count = self.scenario_states['retry_count'].get(i, 0)
                status_text += f" ❌ ({retry_count}회 재시도)"
            elif i in self.scenario_states['skipped_steps']:
                # 건너뛴 단계
                w.label.setStyleSheet('font-weight: bold; color: orange;')
                status_text += " ⏭️"
            elif i == self.current_step - 1:
                # 현재 진행 중인 단계
                w.label.setStyleSheet('font-weight: bold; color: blue;')
                status_text += " 🔄"
            else:
                # 아직 시작하지 않은 단계
                w.label.setStyleSheet('font-weight: normal; color: gray;')
            
            w.label.setText(status_text)

    def on_webview_load_finished(self, ok):
        if ok:
            self.log("웹뷰 로드 성공!")
            # React 앱이 완전히 로드될 때까지 기다린 후 페이지 내용 확인
            QTimer.singleShot(8000, lambda: self._check_page_content())
        else:
            self.log("웹뷰 로드 실패! (404/네트워크 문제 등)")
            self.log("💡 브라우저에서 직접 http://localhost:3010 접속해보세요")
            # 실패 원인 진단
            current_url = self.webview.url().toString()
            self.log(f"실패한 URL: {current_url}")
            # 네트워크 연결 테스트
            self._test_network_connection()

    def _check_page_content(self):
        """페이지 내용 확인"""
        self.log("페이지 내용 확인 중...")
        # 먼저 제목 확인
        self.webview.page().runJavaScript("document.title", self._on_page_title_received)
        # 그 다음 전체 페이지 구조 확인
        self.webview.page().runJavaScript("""
            (function() {
                return {
                    title: document.title,
                    readyState: document.readyState,
                    bodyContent: document.body ? document.body.innerHTML.substring(0, 500) : 'No body',
                    hasRoot: !!document.getElementById('root'),
                    rootContent: document.getElementById('root') ? document.getElementById('root').innerHTML.substring(0, 500) : 'No root element',
                    scripts: Array.from(document.scripts).map(s => s.src).filter(src => src),
                    links: Array.from(document.links).map(l => l.href).filter(href => href)
                };
            })()
        """, self._on_full_page_details_received)

    def _on_page_title_received(self, title):
        """페이지 제목 수신 시 호출"""
        if title:
            self.log(f"페이지 제목: {title}")
        else:
            self.log("페이지 제목을 가져올 수 없습니다.")
            # 더 자세한 페이지 정보 확인
            self.webview.page().runJavaScript("""
                (function() {
                    return {
                        title: document.title,
                        readyState: document.readyState,
                        bodyContent: document.body ? document.body.innerHTML.substring(0, 200) : 'No body',
                        hasRoot: !!document.getElementById('root'),
                        rootContent: document.getElementById('root') ? document.getElementById('root').innerHTML.substring(0, 200) : 'No root element'
                    };
                })()
            """, self._on_page_details_received)

    def _on_full_page_details_received(self, details):
        """전체 페이지 상세 정보 수신 시 호출"""
        if details:
            self.log(f"📄 전체 페이지 상세 정보:")
            self.log(f"  - 제목: {details.get('title', 'N/A')}")
            self.log(f"  - 준비상태: {details.get('readyState', 'N/A')}")
            self.log(f"  - root 요소 존재: {details.get('hasRoot', False)}")
            self.log(f"  - root 내용: {details.get('rootContent', 'N/A')}")
            self.log(f"  - 스크립트 파일: {details.get('scripts', [])}")
            self.log(f"  - 링크 파일: {details.get('links', [])}")
            
            # root 요소가 없으면 문제 진단
            if not details.get('hasRoot', False):
                self.log("❌ root 요소가 없습니다. React 앱이 제대로 로드되지 않았습니다.")
                self.log("💡 브라우저에서 직접 접속해서 확인해보세요.")
        else:
            self.log("❌ 전체 페이지 상세 정보를 가져올 수 없습니다.")

    def _on_page_details_received(self, details):
        """페이지 상세 정보 수신 시 호출"""
        if details:
            self.log(f"📄 페이지 상세 정보:")
            self.log(f"  - 제목: {details.get('title', 'N/A')}")
            self.log(f"  - 준비상태: {details.get('readyState', 'N/A')}")
            self.log(f"  - root 요소 존재: {details.get('hasRoot', False)}")
            self.log(f"  - root 내용: {details.get('rootContent', 'N/A')}")
        else:
            self.log("❌ 페이지 상세 정보를 가져올 수 없습니다.")

    def test_external_site(self):
        """외부 사이트 테스트"""
        self.log("🌐 네이버 테스트 시작...")
        url = QUrl("https://www.naver.com")
        self.webview.setUrl(url)
        self.log(f"네이버 로드 시도: {url.toString()}")
        
        # 5초 후 로드 상태 확인
        QTimer.singleShot(5000, lambda: self._check_external_site())

    def _check_external_site(self):
        """외부 사이트 로드 상태 확인"""
        self.log("🔍 외부 사이트 로드 상태 확인...")
        self.webview.page().runJavaScript("""
            (function() {
                return {
                    title: document.title,
                    readyState: document.readyState,
                    bodyContent: document.body ? document.body.innerHTML.substring(0, 200) : 'No body',
                    hasContent: document.body && document.body.innerHTML.length > 0
                };
            })()
        """, self._on_external_site_result)

    def _on_external_site_result(self, result):
        """외부 사이트 로드 결과"""
        if result:
            self.log(f"🌐 외부 사이트 로드 결과:")
            self.log(f"  - 제목: {result.get('title', 'N/A')}")
            self.log(f"  - 준비상태: {result.get('readyState', 'N/A')}")
            self.log(f"  - 내용 존재: {result.get('hasContent', False)}")
            
            if result.get('hasContent', False):
                self.log("✅ 외부 사이트 로드 성공! 웹뷰는 정상 작동합니다.")
                self.log("💡 localhost 문제일 가능성이 높습니다.")
            else:
                self.log("❌ 외부 사이트도 로드되지 않습니다. 웹뷰 자체에 문제가 있습니다.")
        else:
            self.log("❌ 외부 사이트 로드 결과를 가져올 수 없습니다.")

    def _test_network_connection(self):
        """네트워크 연결 테스트"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', 3010))
            sock.close()
            
            if result == 0:
                self.log("✅ 127.0.0.1:3010 연결 성공")
            else:
                self.log("❌ 127.0.0.1:3010 연결 실패")
        except Exception as e:
            self.log(f"❌ 네트워크 테스트 오류: {e}")

    def check_scenario_step_completion(self, clicked_element):
        """클릭된 요소가 현재 시나리오 단계를 완료하는지 확인"""
        if self.current_step >= len(self.scenario_step_widgets) or self.scenario_completed:
            return False
            
        current_step_text = self.scenario_step_widgets[self.current_step].edit.text().strip()
        
        # 클릭된 요소의 정보 가져오기
        element_info = {
            'tagName': clicked_element.get('tagName', ''),
            'className': clicked_element.get('className', ''),
            'textContent': clicked_element.get('textContent', ''),
            'id': clicked_element.get('id', ''),
            'dataName': clicked_element.get('dataName', '')
        }
        
        # 행동 기록
        self.record_action('click', element_info)
        
        # 시선추적 데이터가 활성화된 경우 클릭 이벤트 기록
        if self.scenario_started and self.tracking_active:
            try:
                import pyautogui
                x, y = pyautogui.position()
                self.record_click_event(x, y, element_info)
            except ImportError:
                self.log("⚠️ pyautogui가 설치되지 않아 마우스 위치를 가져올 수 없습니다.")
                # 기본값으로 클릭 이벤트 기록
                self.record_click_event(0, 0, element_info)
        
        self.log(f"클릭된 요소: {element_info}")
        self.log(f"현재 단계: {current_step_text}")
        
        # 취소/뒤로가기 버튼 처리
        if self.is_cancel_action(element_info):
            self.handle_cancel_action()
            return False
        
        # 시나리오 단계별 완료 조건 확인
        if current_step_text == "아메리카노":
            return self.check_americano_completion(element_info)
        elif current_step_text == "결제":
            return self.check_payment_completion(element_info)
        else:
            return self.check_general_menu_completion(element_info, current_step_text)

    def is_cancel_action(self, element_info):
        """취소/뒤로가기 행동인지 확인"""
        text_content = element_info['textContent'].strip().lower()
        class_name = element_info['className'].lower()
        
        cancel_keywords = ['취소', '뒤로', 'back', 'cancel', 'close']
        cancel_classes = ['btn-cancel', 'btn-back', 'cancel', 'back']
        
        return (any(keyword in text_content for keyword in cancel_keywords) or
                any(cls in class_name for cls in cancel_classes))

    def handle_cancel_action(self):
        """취소 행동 처리"""
        self.log("⚠️ 사용자가 취소/뒤로가기 버튼을 클릭했습니다.")
        
        # 현재 단계를 실패로 처리하거나 건너뛰기
        if self.current_step < len(self.scenario_step_widgets):
            current_step_text = self.scenario_step_widgets[self.current_step].edit.text().strip()
            
            # 필수 단계인 경우 실패로 처리
            if current_step_text in ["아메리카노", "결제"]:
                self.fail_current_step("사용자가 취소함")
            else:
                # 선택적 단계인 경우 건너뛰기
                self.skip_current_step("사용자가 취소함")

    def check_americano_completion(self, element_info):
        """아메리카노 선택 완료 조건 확인"""
        text_content = element_info['textContent'].strip()
        element_id = element_info['id'].lower()
        data_name = element_info['dataName'].strip()
        
        # 아메리카노 텍스트가 포함된 경우
        if '아메리카노' in text_content:
            self.log("✅ 아메리카노 선택 완료! (텍스트 매칭)")
            return True
        
        # 아메리카노 이미지 ID인 경우 (cof_ame)
        if element_id == 'cof_ame':
            self.log("✅ 아메리카노 선택 완료! (이미지 ID 매칭)")
            return True
        
        # data-name 속성이 아메리카노인 경우
        if data_name == '아메리카노':
            self.log("✅ 아메리카노 선택 완료! (data-name 매칭)")
            return True
        
        # 다른 메뉴를 선택한 경우
        if element_id in ['cof_lat', 'cof_cap', 'cof_cm', 'cof_moca', 'cof_cb', 'cof_goodhazel', 'cof_black']:
            self.log(f"⚠️ 아메리카노 대신 다른 커피를 선택했습니다: {data_name or element_id}")
            self.fail_current_step(f"잘못된 메뉴 선택: {data_name or element_id}")
            return False
        
        return False

    def check_payment_completion(self, element_info):
        """결제 완료 조건 확인"""
        text_content = element_info['textContent'].strip()
        class_name = element_info['className'].lower()
        
        # 결제 관련 텍스트가 포함된 경우
        if any(keyword in text_content for keyword in ['결제', '카드', '현금', 'Pay', 'Checkout']):
            self.log("✅ 결제 완료! (결제 텍스트)")
            return True
        
        # 결제 버튼 클래스가 있는 경우
        if any(keyword in class_name for keyword in ['btn-primary', 'btn-pay', 'checkout', 'payment']):
            self.log("✅ 결제 완료! (결제 버튼 클래스)")
            return True
        
        return False

    def check_general_menu_completion(self, element_info, current_step_text):
        """일반적인 메뉴 선택 완료 조건 확인"""
        element_id = element_info['id'].lower()
        data_name = element_info['dataName'].strip()
        text_content = element_info['textContent'].strip()
        
        # 메뉴 이미지 ID가 있는 경우 (모든 메뉴의 imageKey)
        menu_ids = [
            'cof_ame', 'cof_lat', 'cof_cap', 'cof_cm', 'cof_moca', 'cof_cb', 'cof_goodhazel', 'cof_black',
            'bev_vanilla', 'bev_caramelcrunch', 'bev_sheercreamlatte', 'cof_bs',
            'nc_choco', 'nc_mint', 'nc_chai', 'nc_greent', 'nc_sweet', 'nc_black',
            'tea_cam', 'tea_hib', 'ade_lemon', 'ade_grape', 'ade_mus', 'tea_peach', 
            'ade_pine_lavender', 'ade_peach_lavender',
            'des_ccake', 'des_tira', 'des_mac', 'des_brown', 'des_yog', 'des_cream',
            'bak_bagel', 'bak_crois', 'bak_sand', 'bak_croff', 'bak_pretz', 'bak_egg',
            'frp_java', 'frp_mocha', 'frp_match', 'frp_straw', 'frp_caram', 'frp_cookie',
            'frp_milkshake', 'frp_coffeemilkshake',
            'sm_berry', 'sm_mango', 'sm_blue', 'sm_peach', 'sm_water', 'sm_kiwi',
            'sm_orange', 'sm_grape', 'sm_grapefruit', 'sm_pineapple'
        ]
        
        if element_id in menu_ids:
            menu_name = data_name if data_name else f"메뉴 (ID: {element_id})"
            self.log(f"✅ {menu_name} 선택 완료! (이미지 ID: {element_id})")
            return True
        
        # 메뉴 텍스트가 있는 경우
        if text_content and len(text_content) > 0:
            self.log(f"✅ {text_content} 선택 완료! (텍스트 매칭)")
            return True
        
        return False

    def check_cart_based_scenario_completion(self):
        """장바구니 상태를 기반으로 시나리오 단계 완료 여부 확인"""
        if self.current_step >= len(self.scenario_step_widgets):
            return False
            
        current_step_text = self.scenario_step_widgets[self.current_step].edit.text().strip()
        
        self.log(f"🛒 장바구니 상태 확인 - 현재 단계: {current_step_text}")
        self.log(f"  - 아이템 수: {self.cart_state['itemCount']}")
        self.log(f"  - 총 수량: {self.cart_state['totalItems']}")
        self.log(f"  - 총 가격: {self.cart_state['totalPrice']:,}원")
        self.log(f"  - 아이템 목록: {[item['name'] for item in self.cart_state['items']]}")
        
        # 시나리오 단계별 장바구니 기반 완료 조건
        if current_step_text == "아메리카노":
            # 장바구니에 아메리카노가 있는지 확인
            for item in self.cart_state['items']:
                if '아메리카노' in item['name']:
                    self.log("✅ 아메리카노가 장바구니에 추가됨!")
                    return True
            
            # 장바구니에 아이템이 있지만 아메리카노가 아닌 경우
            if self.cart_state['itemCount'] > 0:
                self.log(f"⚠️ 장바구니에 다른 아이템이 있습니다: {[item['name'] for item in self.cart_state['items']]}")
                # 다른 메뉴도 허용하도록 수정
                return True
                
        elif current_step_text == "결제":
            # 장바구니에 아이템이 있고 결제 단계에 도달했는지 확인
            if self.cart_state['itemCount'] > 0 and self.cart_state['step'] >= 6:
                self.log("✅ 결제 단계 도달!")
                return True
                
        elif current_step_text == "장바구니 담기":
            # 장바구니에 아이템이 추가되었는지 확인
            if self.cart_state['itemCount'] > 0:
                self.log("✅ 장바구니에 아이템이 담겼습니다!")
                return True
                
        elif current_step_text == "장바구니 확인":
            # 장바구니 화면에 도달했는지 확인
            if self.cart_state['step'] == 5:
                self.log("✅ 장바구니 화면 확인!")
                return True
                
        elif current_step_text == "주문 완료":
            # 주문이 완료되었는지 확인 (step 9)
            if self.cart_state['step'] == 9:
                self.log("✅ 주문 완료!")
                return True
        
        # 일반적인 메뉴 선택 (특정 메뉴명이 단계명인 경우)
        else:
            # 장바구니에 해당 메뉴가 있는지 확인
            for item in self.cart_state['items']:
                if current_step_text in item['name']:
                    self.log(f"✅ {current_step_text}가 장바구니에 추가됨!")
                    return True
        
        return False

    def handle_cart_change(self, cart_data):
        """장바구니 상태 변경 처리"""
        import json
        
        try:
            if isinstance(cart_data, str):
                cart_data = json.loads(cart_data)
            
            # 이전 상태 저장
            previous_state = self.cart_state.copy()
            
            # 새로운 상태 업데이트
            self.cart_state.update(cart_data)
            
            # 변경 이력에 추가
            self.cart_history.append({
                'timestamp': time.time(),
                'previous': previous_state,
                'current': self.cart_state.copy()
            })
            
            # 행동 기록
            self.record_action('cart_change', {
                'previous': previous_state,
                'current': self.cart_state.copy()
            })
            
            self.log(f"🛒 장바구니 상태 변경:")
            self.log(f"  - 아이템 수: {previous_state['itemCount']} → {self.cart_state['itemCount']}")
            self.log(f"  - 총 수량: {previous_state['totalItems']} → {self.cart_state['totalItems']}")
            self.log(f"  - 총 가격: {previous_state['totalPrice']:,}원 → {self.cart_state['totalPrice']:,}원")
            self.log(f"  - 현재 단계: {self.cart_state['step']}")
            
            # 장바구니 비우기 감지
            if previous_state['itemCount'] > 0 and self.cart_state['itemCount'] == 0:
                self.handle_cart_clear()
                return
            
            # 장바구니 기반 시나리오 완료 확인
            if self.check_cart_based_scenario_completion():
                self.advance_scenario_step()
                
        except Exception as e:
            self.log(f"❌ 장바구니 상태 처리 오류: {e}")

    def handle_cart_clear(self):
        """장바구니 비우기 처리"""
        self.log("🗑️ 장바구니가 비워졌습니다.")
        
        if self.current_step < len(self.scenario_step_widgets):
            current_step_text = self.scenario_step_widgets[self.current_step].edit.text().strip()
            
            # 장바구니 담기 단계에서 장바구니를 비운 경우
            if current_step_text in ["아메리카노", "장바구니 담기"]:
                self.fail_current_step("장바구니를 비워서 취소함")
            elif current_step_text == "결제":
                # 결제 단계에서 장바구니를 비운 경우
                self.log("⚠️ 결제 단계에서 장바구니를 비웠습니다. 다시 메뉴 선택으로 돌아갑니다.")
                # 이전 단계로 되돌리기 (메뉴 선택 단계)
                self.rollback_to_menu_selection()

    def rollback_to_menu_selection(self):
        """메뉴 선택 단계로 되돌리기"""
        # 아메리카노 단계(첫 번째 단계)로 되돌리기
        target_step = 0
        
        # 현재 단계부터 목표 단계까지의 실패 기록 제거
        for step in range(self.current_step, target_step, -1):
            if step in self.scenario_states['failed_steps']:
                self.scenario_states['failed_steps'].remove(step)
            if step in self.scenario_states['retry_count']:
                del self.scenario_states['retry_count'][step]
        
        # 현재 단계를 목표 단계로 설정
        self.current_step = target_step
        self.scenario_states['current_step'] = target_step
        
        self.update_scenario_ui()
        self.log(f"🔄 메뉴 선택 단계로 되돌렸습니다. (단계 {self.current_step + 1})")

    def handle_web_message(self, message_data):
        """웹뷰에서 온 메시지 처리"""
        import json
        
        try:
            if isinstance(message_data, str):
                message_data = json.loads(message_data)
            
            message_type = message_data.get('type', '')
            
            if message_type == 'cartChange':
                self.handle_cart_change(message_data.get('data', {}))
            elif message_type == 'click':
                element_info = message_data.get('data', {})
                if self.check_scenario_step_completion(element_info):
                    self.advance_scenario_step()
            elif message_type == 'pageLoaded':
                self.log(f"📄 페이지 로드 완료: {message_data.get('data', {}).get('url', '')}")
            else:
                self.log(f"📨 알 수 없는 메시지 타입: {message_type}")
                
        except Exception as e:
            self.log(f"❌ 웹 메시지 처리 오류: {e}")

    def get_screen_resolution(self):
        """화면 해상도를 가져옵니다."""
        if EYEWARE_AVAILABLE:
            try:
                primary_monitor = next((m for m in get_monitors() if m.is_primary), get_monitors()[0])
                return primary_monitor.width, primary_monitor.height
            except:
                pass
        return 1920, 1080

    def update_eye_tracking_status(self, active=False):
        """시선추적 상태를 업데이트합니다."""
        if active:
            self.eye_tracking_status.setText("👁️ 시선추적: 활성")
            self.eye_tracking_status.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.eye_tracking_status.setText("👁️ 시선추적: 비활성")
            self.eye_tracking_status.setStyleSheet("color: gray; font-weight: bold;")

    def update_data_collection_status(self, collecting=False):
        """데이터 수집 상태를 업데이트합니다."""
        if collecting:
            self.data_collection_status.setText("📊 데이터 수집: 수집 중")
            self.data_collection_status.setStyleSheet("color: blue; font-weight: bold;")
        else:
            self.data_collection_status.setText("📊 데이터 수집: 대기 중")
            self.data_collection_status.setStyleSheet("color: gray; font-weight: bold;")

    def start_eye_tracker(self):
        """시선추적을 시작합니다."""
        if not EYEWARE_AVAILABLE:
            self.log("❌ Eyeware SDK를 사용할 수 없습니다.")
            return False
            
        try:
            viewport = bet.ViewportGeometry()
            viewport.point_00.x = 0
            viewport.point_00.y = 0
            viewport.point_11.x = self.screen_width
            viewport.point_11.y = self.screen_height
            
            self.eye_tracker = bet.API("uiux-certification-app", viewport)
            self.eye_tracker.attempt_starting_the_beam_eye_tracker()
            self.tracking_active = True
            self.last_update_timestamp = bet.NULL_DATA_TIMESTAMP()
            
            # 시선추적 스레드 시작
            self.start_gaze_thread()
            
            # 상태 업데이트
            self.update_eye_tracking_status(True)
            self.update_data_collection_status(True)
            
            self.log("✅ 시선추적이 시작되었습니다.")
            return True
            
        except Exception as e:
            self.log(f"❌ 시선추적 시작 실패: {e}")
            return False

    def start_gaze_thread(self):
        """시선추적 데이터 수집 스레드를 시작합니다."""
        if not self.eye_tracker:
            return
            
        self.gaze_thread = threading.Thread(target=self._gaze_loop)
        self.gaze_thread.daemon = True
        self.gaze_thread.start()
        self.log("👁️ 시선추적 스레드가 시작되었습니다.")

    def _gaze_loop(self):
        """시선추적 데이터를 지속적으로 수집합니다."""
        while self.tracking_active:
            try:
                if self.eye_tracker.wait_for_new_tracking_state_set(self.last_update_timestamp, 100):
                    tracking_state_set = self.eye_tracker.get_latest_tracking_state_set()
                    user_state = tracking_state_set.user_state()
                    
                    if (user_state.unified_screen_gaze.confidence != bet.TrackingConfidence.LOST_TRACKING and
                        user_state.timestamp_in_seconds != bet.NULL_DATA_TIMESTAMP()):
                        
                        gaze = user_state.unified_screen_gaze.point_of_regard
                        gaze_data_point = {
                            'timestamp': time.time(),
                            'x': gaze.x,
                            'y': gaze.y,
                            'confidence': user_state.unified_screen_gaze.confidence,
                            'step': self.current_step
                        }
                        
                        self.gaze_data.append(gaze_data_point)
                        self.current_gaze_segment.append(gaze_data_point)
                        
                    self.last_update_timestamp = user_state.timestamp_in_seconds
                    
            except Exception as e:
                self.log(f"❌ 시선추적 데이터 수집 오류: {e}")
                break
                
            time.sleep(0.01)

    def take_screenshot(self, filename=None):
        """웹뷰(QWebEngineView) 영역만 정확히 캡처합니다."""
        try:
            if filename is None:
                filename = f"screenshot_{len(self.click_events)+1}.png"
            screenshot_path = os.path.join(self.scenario_data_dir, filename)
            # QWebEngineView의 grab() 사용
            pixmap = self.webview.grab()
            pixmap.save(screenshot_path)
            self.current_screenshot_path = screenshot_path
            self.log(f"📸 웹뷰 스크린샷 저장: {filename} (QWebEngineView.grab)")
            return screenshot_path
        except Exception as e:
            self.log(f"❌ 스크린샷 촬영 실패: {e}")
            return None

    def create_heatmap(self, gaze_points, screenshot_path, output_filename=None):
        """시선 히트맵을 생성하고 저장합니다."""
        if not gaze_points or not os.path.exists(screenshot_path):
            return None
            
        try:
            if output_filename is None:
                output_filename = f"heatmap_{len(self.click_events)}.png"
            
            output_path = os.path.join(self.scenario_data_dir, output_filename)
            
            # 스크린샷 로드
            img = Image.open(screenshot_path)
            img_array = np.array(img)
            img_height = img_array.shape[0]
            
            # 웹뷰의 글로벌 좌상단 좌표
            webview_pos = self.webview.mapToGlobal(self.webview.rect().topLeft())
            webview_x, webview_y = webview_pos.x(), webview_pos.y()
            
            # 시선 데이터 플롯 (웹뷰 기준 상대좌표로 변환, y축 뒤집기)
            x_coords = [g['x'] - webview_x for g in gaze_points if 'x' in g and 'y' in g]
            y_coords = [img_height - (g['y'] - webview_y) for g in gaze_points if 'x' in g and 'y' in g]
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img_array, origin='upper', zorder=1)
            
            if x_coords and y_coords:
                h = plt.hist2d(x_coords, y_coords, bins=30, cmap='hot', alpha=0.5, zorder=2)
                plt.colorbar(h[3], label='시선 빈도')
                plt.plot(x_coords, y_coords, 'b-', alpha=0.3, linewidth=1, zorder=3)
                plt.scatter(x_coords, y_coords, c='blue', s=10, alpha=0.5, zorder=4)
            
            plt.title(f'시선 히트맵 - 클릭 {len(self.click_events)}')
            plt.xlabel('X 좌표 (픽셀)')
            plt.ylabel('Y 좌표 (픽셀)')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            self.log(f"🔥 히트맵 저장: {output_filename}")
            return output_path
            
        except Exception as e:
            self.log(f"❌ 히트맵 생성 실패: {e}")
            return None

    def record_click_event(self, x, y, element_info=None):
        """클릭 이벤트를 기록합니다."""
        click_event = {
            'timestamp': time.time(),
            'x': x,
            'y': y,
            'step': self.current_step,
            'element_info': element_info,
            'screenshot_path': self.current_screenshot_path,
            'gaze_count': len(self.current_gaze_segment)
        }
        
        self.click_events.append(click_event)
        self.log(f"🖱️ 클릭 기록: ({x}, {y}) - 단계 {self.current_step}")
        
        # 현재 시선 세그먼트로 히트맵 생성
        if self.current_gaze_segment and self.current_screenshot_path:
            heatmap_filename = f"heatmap_click_{len(self.click_events)}.png"
            self.create_heatmap(self.current_gaze_segment, self.current_screenshot_path, heatmap_filename)
        
        # 새로운 스크린샷 촬영 및 시선 세그먼트 초기화
        self.take_screenshot(f"screenshot_click_{len(self.click_events)+1}.png")
        self.current_gaze_segment = []

    def save_scenario_data(self):
        """시나리오 데이터를 저장합니다."""
        try:
            # 시선 데이터 저장
            gaze_data_path = os.path.join(self.scenario_data_dir, "gaze_data.json")
            with open(gaze_data_path, 'w', encoding='utf-8') as f:
                json.dump(self.gaze_data, f, ensure_ascii=False, indent=2)
            
            # 클릭 이벤트 저장
            click_events_path = os.path.join(self.scenario_data_dir, "click_events.json")
            with open(click_events_path, 'w', encoding='utf-8') as f:
                json.dump(self.click_events, f, ensure_ascii=False, indent=2)
            
            # 시나리오 데이터 저장
            scenario_data = {
                'timestamp': self.scenario_timestamp,
                'scenario_steps': [w.edit.text().strip() for w in self.scenario_step_widgets],
                'scenario_states': {
                    'completed_steps': list(self.scenario_states['completed_steps']),
                    'failed_steps': list(self.scenario_states['failed_steps']),
                    'skipped_steps': list(self.scenario_states['skipped_steps']),
                    'retry_count': self.scenario_states['retry_count']
                },
                'cart_history': self.cart_history,
                'final_cart_state': self.cart_state,
                'total_gaze_points': len(self.gaze_data),
                'total_clicks': len(self.click_events)
            }
            
            scenario_data_path = os.path.join(self.scenario_data_dir, "scenario_data.json")
            with open(scenario_data_path, 'w', encoding='utf-8') as f:
                json.dump(scenario_data, f, ensure_ascii=False, indent=2)
            
            self.log(f"💾 시나리오 데이터 저장 완료: {self.scenario_data_dir}")
            
        except Exception as e:
            self.log(f"❌ 시나리오 데이터 저장 실패: {e}")

    def stop_eye_tracker(self):
        """시선추적을 중지합니다."""
        self.tracking_active = False
        if self.gaze_thread:
            self.gaze_thread.join(timeout=1.0)
        
        if self.eye_tracker:
            try:
                self.eye_tracker.stop_the_beam_eye_tracker()
                self.log("🛑 시선추적이 중지되었습니다.")
            except:
                pass
        
        # 상태 업데이트
        self.update_eye_tracking_status(False)
        self.update_data_collection_status(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_()) 