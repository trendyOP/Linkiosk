import cv2
import numpy as np
import json
import time
import threading
from typing import Dict, List, Tuple, Optional
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
import sys
import os

# Beam Eye Tracker SDK import (실제 SDK 경로에 맞게 수정 필요)
try:
    import eyeware
except ImportError:
    print("Beam Eye Tracker SDK를 찾을 수 없습니다. 시뮬레이션 모드로 실행됩니다.")
    eyeware = None

class GazeTracker:
    """시선 추적기 클래스"""
    
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.gaze_data = []
        self.is_tracking = False
        self.current_gaze = {'x': 0, 'y': 0, 'confidence': 0}
        
        if not simulation_mode and eyeware:
            self.init_beam_tracker()
        else:
            self.init_simulation()
    
    def init_beam_tracker(self):
        """Beam Eye Tracker 초기화"""
        try:
            # Beam Eye Tracker SDK 초기화 코드
            # 실제 SDK 문서에 따라 구현
            pass
        except Exception as e:
            print(f"Beam Eye Tracker 초기화 실패: {e}")
            self.simulation_mode = True
            self.init_simulation()
    
    def init_simulation(self):
        """시뮬레이션 모드 초기화"""
        print("시선 추적 시뮬레이션 모드 활성화")
        self.simulation_mode = True
    
    def start_tracking(self):
        """시선 추적 시작"""
        self.is_tracking = True
        if self.simulation_mode:
            self.simulation_thread = threading.Thread(target=self._simulate_gaze)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
    
    def stop_tracking(self):
        """시선 추적 중지"""
        self.is_tracking = False
    
    def _simulate_gaze(self):
        """시선 추적 시뮬레이션"""
        import random
        
        while self.is_tracking:
            # 랜덤한 시선 좌표 생성 (화면 중앙 주변)
            x = random.randint(400, 1200)
            y = random.randint(300, 800)
            confidence = random.uniform(0.7, 1.0)
            
            self.current_gaze = {
                'x': x,
                'y': y,
                'confidence': confidence,
                'timestamp': time.time()
            }
            
            self.gaze_data.append(self.current_gaze.copy())
            time.sleep(0.033)  # 30 FPS
    
    def get_current_gaze(self) -> Dict:
        """현재 시선 좌표 반환"""
        return self.current_gaze
    
    def get_gaze_history(self) -> List[Dict]:
        """시선 추적 히스토리 반환"""
        return self.gaze_data

class AdaptiveUIWidget(QWidget):
    """적응형 UI 위젯"""
    
    def __init__(self, user_type: str = "elderly"):
        super().__init__()
        self.user_type = user_type
        self.ui_settings = self.get_ui_settings(user_type)
        self.gaze_tracker = GazeTracker(simulation_mode=True)
        self.gaze_history = []
        self.fixation_areas = []
        self.adaptive_buttons = {}  # 버튼 참조 저장
        self.fixation_threshold = 1.0  # 고정 시간 임계값 (초)
        self.zoom_factor = 1.2  # 확대 배율
        
        self.init_ui()
        self.start_gaze_tracking()
    
    def get_ui_settings(self, user_type: str) -> Dict:
        """사용자 유형에 따른 UI 설정"""
        settings = {
            'young': {
                'font_size': 16,
                'button_height': 40,
                'button_width': 120,
                'text_color': '#333333',
                'background_color': '#ffffff',
                'button_color': '#007bff',
                'button_hover_color': '#0056b3',
                'spacing': 10
            },
            'middle': {
                'font_size': 18,
                'button_height': 50,
                'button_width': 140,
                'text_color': '#222222',
                'background_color': '#f8f9fa',
                'button_color': '#28a745',
                'button_hover_color': '#1e7e34',
                'spacing': 15
            },
            'elderly': {
                'font_size': 24,
                'button_height': 70,
                'button_width': 180,
                'text_color': '#000000',
                'background_color': '#ffffff',
                'button_color': '#dc3545',
                'button_hover_color': '#c82333',
                'spacing': 20
            }
        }
        return settings.get(user_type, settings['elderly'])
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle(f"적응형 키오스크 UI ({self.user_type})")
        self.setGeometry(100, 100, 1200, 800)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        
        # 상단 정보 패널
        info_panel = self.create_info_panel()
        main_layout.addWidget(info_panel)
        
        # 메뉴 버튼들
        menu_layout = QGridLayout()
        menu_items = [
            "아메리카노", "카페라떼", "카푸치노", "에스프레소",
            "수박주스", "오렌지주스", "사과주스", "포도주스",
            "햄&치즈 샌드위치", "치킨 샌드위치", "베지 샌드위치", "터키 샌드위치"
        ]
        
        row, col = 0, 0
        for item in menu_items:
            button = self.create_adaptive_button(item)
            menu_layout.addWidget(button, row, col)
            self.adaptive_buttons[item] = button  # 버튼 참조 저장
            col += 1
            if col >= 4:
                col = 0
                row += 1
        
        menu_widget = QWidget()
        menu_widget.setLayout(menu_layout)
        main_layout.addWidget(menu_widget)
        
        # 하단 컨트롤 패널
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        self.setLayout(main_layout)
        self.apply_ui_settings()
    
    def create_info_panel(self) -> QWidget:
        """정보 패널 생성"""
        panel = QWidget()
        panel.setMaximumHeight(100)
        panel.setStyleSheet(f"background-color: {self.ui_settings['background_color']}; border: 2px solid #ddd;")
        
        layout = QHBoxLayout()
        
        # 사용자 유형 표시
        user_label = QLabel(f"사용자 유형: {self.user_type.upper()}")
        user_label.setStyleSheet(f"font-size: {self.ui_settings['font_size']}px; color: {self.ui_settings['text_color']}; font-weight: bold;")
        layout.addWidget(user_label)
        
        # 시선 추적 상태
        self.gaze_status_label = QLabel("시선 추적: 비활성")
        self.gaze_status_label.setStyleSheet(f"font-size: {self.ui_settings['font_size']-4}px; color: {self.ui_settings['text_color']};")
        layout.addWidget(self.gaze_status_label)
        
        # 현재 시선 좌표
        self.gaze_coord_label = QLabel("시선 좌표: (0, 0)")
        self.gaze_coord_label.setStyleSheet(f"font-size: {self.ui_settings['font_size']-4}px; color: {self.ui_settings['text_color']};")
        layout.addWidget(self.gaze_coord_label)
        
        # 적응형 UI 상태
        self.adaptive_status_label = QLabel("적응형 UI: 비활성")
        self.adaptive_status_label.setStyleSheet(f"font-size: {self.ui_settings['font_size']-4}px; color: {self.ui_settings['text_color']};")
        layout.addWidget(self.adaptive_status_label)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_adaptive_button(self, text: str) -> QPushButton:
        """적응형 버튼 생성"""
        button = QPushButton(text)
        button.setMinimumSize(self.ui_settings['button_width'], self.ui_settings['button_height'])
        button.setStyleSheet(f"""
            QPushButton {{
                font-size: {self.ui_settings['font_size']}px;
                font-weight: bold;
                color: white;
                background-color: {self.ui_settings['button_color']};
                border: 2px solid {self.ui_settings['button_color']};
                border-radius: 10px;
                padding: 10px;
            }}
            QPushButton:hover {{
                background-color: {self.ui_settings['button_hover_color']};
                border-color: {self.ui_settings['button_hover_color']};
            }}
            QPushButton:pressed {{
                background-color: #666666;
            }}
        """)
        
        # 버튼 클릭 이벤트
        button.clicked.connect(lambda: self.on_button_click(text))
        
        return button
    
    def create_control_panel(self) -> QWidget:
        """컨트롤 패널 생성"""
        panel = QWidget()
        panel.setMaximumHeight(80)
        panel.setStyleSheet(f"background-color: {self.ui_settings['background_color']}; border: 2px solid #ddd;")
        
        layout = QHBoxLayout()
        
        # 시선 추적 시작/중지 버튼
        self.tracking_button = QPushButton("시선 추적 시작")
        self.tracking_button.setStyleSheet(f"""
            QPushButton {{
                font-size: {self.ui_settings['font_size']-4}px;
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: #218838;
            }}
        """)
        self.tracking_button.clicked.connect(self.toggle_gaze_tracking)
        layout.addWidget(self.tracking_button)
        
        # UI 설정 변경 버튼
        settings_button = QPushButton("UI 설정 변경")
        settings_button.setStyleSheet(f"""
            QPushButton {{
                font-size: {self.ui_settings['font_size']-4}px;
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: #138496;
            }}
        """)
        settings_button.clicked.connect(self.change_ui_settings)
        layout.addWidget(settings_button)
        
        # 분석 결과 표시 버튼
        analysis_button = QPushButton("시선 분석")
        analysis_button.setStyleSheet(f"""
            QPushButton {{
                font-size: {self.ui_settings['font_size']-4}px;
                background-color: #ffc107;
                color: black;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
            }}
            QPushButton:hover {{
                background-color: #e0a800;
            }}
        """)
        analysis_button.clicked.connect(self.show_gaze_analysis)
        layout.addWidget(analysis_button)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def apply_ui_settings(self):
        """UI 설정 적용"""
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.ui_settings['background_color']};
                color: {self.ui_settings['text_color']};
                font-size: {self.ui_settings['font_size']}px;
            }}
        """)
    
    def start_gaze_tracking(self):
        """시선 추적 시작"""
        self.gaze_tracker.start_tracking()
        self.gaze_timer = QTimer()
        self.gaze_timer.timeout.connect(self.update_gaze_display)
        self.gaze_timer.start(100)  # 10 FPS로 업데이트
    
    def toggle_gaze_tracking(self):
        """시선 추적 토글"""
        if self.gaze_tracker.is_tracking:
            self.gaze_tracker.stop_tracking()
            self.tracking_button.setText("시선 추적 시작")
            self.gaze_status_label.setText("시선 추적: 비활성")
        else:
            self.gaze_tracker.start_tracking()
            self.tracking_button.setText("시선 추적 중지")
            self.gaze_status_label.setText("시선 추적: 활성")
    
    def update_gaze_display(self):
        """시선 표시 업데이트"""
        gaze = self.gaze_tracker.get_current_gaze()
        self.gaze_coord_label.setText(f"시선 좌표: ({gaze['x']}, {gaze['y']})")
        
        # 시선 히스토리 업데이트
        self.gaze_history = self.gaze_tracker.get_gaze_history()
        
        # 고정점 분석
        self.analyze_fixations()
        
        # 적응형 UI 업데이트
        self.update_adaptive_ui(gaze)
    
    def analyze_fixations(self):
        """고정점 분석"""
        if len(self.gaze_history) < 10:
            return
        
        # 최근 30개 시선 점으로 고정점 분석
        recent_gaze = self.gaze_history[-30:]
        
        # 고정점 감지 (30픽셀 반경 내에서 0.5초 이상 머무름)
        fixations = []
        current_fixation = None
        
        for gaze in recent_gaze:
            if current_fixation is None:
                current_fixation = {
                    'start_time': gaze['timestamp'],
                    'center_x': gaze['x'],
                    'center_y': gaze['y'],
                    'points': [gaze]
                }
            else:
                # 현재 점과 고정점 중심점 간의 거리 계산
                center_x = np.mean([p['x'] for p in current_fixation['points']])
                center_y = np.mean([p['y'] for p in current_fixation['points']])
                
                distance = np.sqrt((gaze['x'] - center_x)**2 + (gaze['y'] - center_y)**2)
                
                if distance <= 30:  # 30픽셀 반경
                    current_fixation['points'].append(gaze)
                else:
                    # 고정점 종료 및 저장
                    duration = gaze['timestamp'] - current_fixation['start_time']
                    if duration >= 0.5:  # 0.5초 이상
                        current_fixation['duration'] = duration
                        current_fixation['center_x'] = center_x
                        current_fixation['center_y'] = center_y
                        fixations.append(current_fixation)
                    
                    # 새로운 고정점 시작
                    current_fixation = {
                        'start_time': gaze['timestamp'],
                        'center_x': gaze['x'],
                        'center_y': gaze['y'],
                        'points': [gaze]
                    }
        
        self.fixation_areas = fixations
    
    def update_adaptive_ui(self, current_gaze: Dict):
        """시선 기반 적응형 UI 업데이트"""
        if not current_gaze or current_gaze['confidence'] < 0.5:
            self.adaptive_status_label.setText("적응형 UI: 신호 없음")
            return
        
        gaze_x, gaze_y = current_gaze['x'], current_gaze['y']
        adaptive_active = False
        
        gaze_x, gaze_y = current_gaze['x'], current_gaze['y']
        
        # 각 버튼에 대해 시선이 머무는지 확인
        for button_text, button in self.adaptive_buttons.items():
            button_rect = button.geometry()
            button_center_x = button_rect.x() + button_rect.width() // 2
            button_center_y = button_rect.y() + button_rect.height() // 2
            
            # 시선과 버튼 중심점 간의 거리 계산
            distance = ((gaze_x - button_center_x) ** 2 + (gaze_y - button_center_y) ** 2) ** 0.5
            
            # 버튼 반경 (대략적인 크기)
            button_radius = max(button_rect.width(), button_rect.height()) // 2
            
            if distance <= button_radius * 1.5:  # 시선이 버튼 근처에 있음
                # 고정 시간 확인
                fixation_duration = self.check_fixation_duration(gaze_x, gaze_y)
                
                if fixation_duration >= self.fixation_threshold:
                    # 버튼 확대 및 강조
                    self.zoom_button(button, True)
                    adaptive_active = True
                    button.setStyleSheet(f"""
                        QPushButton {{
                            font-size: {int(self.ui_settings['font_size'] * self.zoom_factor)}px;
                            font-weight: bold;
                            color: white;
                            background-color: #ff6b6b;
                            border: 3px solid #ff4757;
                            border-radius: 15px;
                            padding: 15px;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                        }}
                        QPushButton:hover {{
                            background-color: #ff5252;
                            border-color: #ff3742;
                        }}
                    """)
                else:
                    # 약간의 강조만
                    self.zoom_button(button, False)
                    adaptive_active = True
                    button.setStyleSheet(f"""
                        QPushButton {{
                            font-size: {self.ui_settings['font_size']}px;
                            font-weight: bold;
                            color: white;
                            background-color: {self.ui_settings['button_color']};
                            border: 2px solid {self.ui_settings['button_color']};
                            border-radius: 10px;
                            padding: 10px;
                        }}
                        QPushButton:hover {{
                            background-color: {self.ui_settings['button_hover_color']};
                            border-color: {self.ui_settings['button_hover_color']};
                        }}
                    """)
            else:
                # 시선이 멀리 있으면 원래 크기로 복원
                self.zoom_button(button, False)
                # adaptive_active는 False로 유지
                button.setStyleSheet(f"""
                    QPushButton {{
                        font-size: {self.ui_settings['font_size']}px;
                        font-weight: bold;
                        color: white;
                        background-color: {self.ui_settings['button_color']};
                        border: 2px solid {self.ui_settings['button_color']};
                        border-radius: 10px;
                        padding: 10px;
                    }}
                    QPushButton:hover {{
                        background-color: {self.ui_settings['button_hover_color']};
                        border-color: {self.ui_settings['button_hover_color']};
                                            }}
                    """)
        
        # 적응형 UI 상태 업데이트
        if adaptive_active:
            self.adaptive_status_label.setText("적응형 UI: 활성")
        else:
            self.adaptive_status_label.setText("적응형 UI: 대기")
    
    def check_fixation_duration(self, gaze_x: int, gaze_y: int) -> float:
        """특정 위치에서의 고정 시간 계산"""
        if len(self.gaze_history) < 10:
            return 0.0
        
        # 최근 시선 데이터에서 해당 위치 근처에 머문 시간 계산
        recent_gaze = self.gaze_history[-30:]  # 최근 30개 시선 점
        fixation_start = None
        total_duration = 0.0
        
        for gaze in recent_gaze:
            if gaze['confidence'] < 0.5:
                continue
            
            distance = ((gaze['x'] - gaze_x) ** 2 + (gaze['y'] - gaze_y) ** 2) ** 0.5
            
            if distance <= 50:  # 50픽셀 반경 내
                if fixation_start is None:
                    fixation_start = gaze['timestamp']
            else:
                if fixation_start is not None:
                    duration = gaze['timestamp'] - fixation_start
                    total_duration += duration
                    fixation_start = None
        
        return total_duration
    
    def zoom_button(self, button: QPushButton, zoom: bool):
        """버튼 확대/축소"""
        if zoom:
            # 버튼 확대
            current_size = button.size()
            new_width = int(current_size.width() * self.zoom_factor)
            new_height = int(current_size.height() * self.zoom_factor)
            button.setMinimumSize(new_width, new_height)
        else:
            # 원래 크기로 복원
            original_width = self.ui_settings['button_width']
            original_height = self.ui_settings['button_height']
            button.setMinimumSize(original_width, original_height)
    
    def on_button_click(self, button_text: str):
        """버튼 클릭 이벤트"""
        QMessageBox.information(self, "선택", f"'{button_text}'을(를) 선택했습니다!")
        
        # 시선 데이터 저장
        gaze_data = {
            'timestamp': time.time(),
            'button_clicked': button_text,
            'gaze_history': self.gaze_history[-50:],  # 최근 50개 시선 점
            'fixations': self.fixation_areas
        }
        
        # 데이터를 JSON 파일로 저장
        filename = f"gaze_data_{int(time.time())}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(gaze_data, f, ensure_ascii=False, indent=2)
        
        print(f"시선 데이터가 {filename}에 저장되었습니다.")
    
    def change_ui_settings(self):
        """UI 설정 변경"""
        user_types = ["young", "middle", "elderly"]
        current_index = user_types.index(self.user_type)
        next_index = (current_index + 1) % len(user_types)
        self.user_type = user_types[next_index]
        
        # UI 설정 업데이트
        self.ui_settings = self.get_ui_settings(self.user_type)
        self.apply_ui_settings()
        
        # 버튼들 업데이트
        self.update_all_buttons()
        
        QMessageBox.information(self, "UI 설정 변경", f"UI 설정이 {self.user_type} 모드로 변경되었습니다.")
    
    def update_all_buttons(self):
        """모든 버튼 스타일 업데이트"""
        for child in self.findChildren(QPushButton):
            if child != self.tracking_button:
                child.setStyleSheet(f"""
                    QPushButton {{
                        font-size: {self.ui_settings['font_size']}px;
                        font-weight: bold;
                        color: white;
                        background-color: {self.ui_settings['button_color']};
                        border: 2px solid {self.ui_settings['button_color']};
                        border-radius: 10px;
                        padding: 10px;
                    }}
                    QPushButton:hover {{
                        background-color: {self.ui_settings['button_hover_color']};
                        border-color: {self.ui_settings['button_hover_color']};
                    }}
                    QPushButton:pressed {{
                        background-color: #666666;
                    }}
                """)
    
    def show_gaze_analysis(self):
        """시선 분석 결과 표시"""
        if not self.gaze_history:
            QMessageBox.warning(self, "분석 불가", "시선 데이터가 충분하지 않습니다.")
            return
        
        # 간단한 분석 결과
        total_points = len(self.gaze_history)
        valid_points = len([p for p in self.gaze_history if p['confidence'] > 0])
        fixation_count = len(self.fixation_areas)
        
        analysis_text = f"""
시선 분석 결과:
- 총 시선 점 수: {total_points}
- 유효한 시선 점 수: {valid_points}
- 고정점 수: {fixation_count}
- 현재 사용자 유형: {self.user_type}
        """
        
        QMessageBox.information(self, "시선 분석", analysis_text)

class AdaptiveKioskApp(QApplication):
    """적응형 키오스크 애플리케이션"""
    
    def __init__(self, argv):
        super().__init__(argv)
        self.setApplicationName("적응형 키오스크")
        self.setApplicationVersion("1.0")
        
        # 메인 윈도우 생성
        self.main_window = AdaptiveUIWidget(user_type="elderly")
        self.main_window.show()

def main():
    """메인 함수"""
    app = AdaptiveKioskApp(sys.argv)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 