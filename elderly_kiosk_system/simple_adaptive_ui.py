#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
간단하고 확실한 적응형 키오스크 UI
버튼 크기 변경이 확실히 작동하는 버전
"""

import sys
import time
import json
import threading
import random
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class SimpleGazeTracker:
    """간단한 시선 추적기"""
    
    def __init__(self):
        self.is_tracking = False
        self.current_gaze = {'x': 0, 'y': 0, 'confidence': 0.8}
        self.gaze_history = []
    
    def start_tracking(self):
        """시선 추적 시작"""
        self.is_tracking = True
        self.simulation_thread = threading.Thread(target=self._simulate_gaze)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def stop_tracking(self):
        """시선 추적 중지"""
        self.is_tracking = False
    
    def _simulate_gaze(self):
        """시선 추적 시뮬레이션"""
        while self.is_tracking:
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            confidence = random.uniform(0.7, 1.0)
            
            self.current_gaze = {
                'x': x,
                'y': y,
                'confidence': confidence,
                'timestamp': time.time()
            }
            
            self.gaze_history.append(self.current_gaze.copy())
            if len(self.gaze_history) > 50:
                self.gaze_history = self.gaze_history[-50:]
            
            time.sleep(0.1)  # 10 FPS
    
    def get_current_gaze(self):
        """현재 시선 좌표 반환"""
        return self.current_gaze

class SimpleAdaptiveUI(QWidget):
    """간단한 적응형 UI"""
    
    def __init__(self):
        super().__init__()
        self.gaze_tracker = SimpleGazeTracker()
        self.buttons = {}
        self.current_user_type = "elderly"  # 기본값
        
        self.init_ui()
        self.start_gaze_tracking()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("간단한 적응형 키오스크")
        self.setGeometry(100, 100, 1000, 700)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        
        # 상단 정보 패널
        info_panel = self.create_info_panel()
        main_layout.addWidget(info_panel)
        
        # 버튼 영역
        button_area = self.create_button_area()
        main_layout.addWidget(button_area)
        
        # 하단 컨트롤 패널
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)
        
        self.setLayout(main_layout)
    
    def create_info_panel(self):
        """정보 패널 생성"""
        panel = QWidget()
        panel.setMaximumHeight(80)
        panel.setStyleSheet("background-color: #f0f0f0; border: 2px solid #ccc; border-radius: 10px;")
        
        layout = QHBoxLayout()
        
        # 시선 좌표
        self.gaze_label = QLabel("시선 좌표: (0, 0)")
        self.gaze_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        layout.addWidget(self.gaze_label)
        
        # 적응형 UI 상태
        self.adaptive_label = QLabel("적응형 UI: 대기")
        self.adaptive_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        layout.addWidget(self.adaptive_label)
        
        # 사용자 유형
        self.user_type_label = QLabel("사용자 유형: 노인")
        self.user_type_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        layout.addWidget(self.user_type_label)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_button_area(self):
        """버튼 영역 생성"""
        area = QWidget()
        area.setStyleSheet("background-color: white; border: 2px solid #ddd; border-radius: 10px;")
        
        # 버튼들을 담을 스크롤 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 버튼 컨테이너
        button_container = QWidget()
        button_layout = QVBoxLayout()
        
        # 메뉴 항목들
        menu_items = [
            "아메리카노", "카페라떼", "카푸치노", "에스프레소",
            "수박주스", "오렌지주스", "사과주스", "포도주스",
            "햄&치즈 샌드위치", "치킨 샌드위치", "베지 샌드위치", "터키 샌드위치"
        ]
        
        for item in menu_items:
            button = self.create_button(item)
            button_layout.addWidget(button)
            self.buttons[item] = button
        
        button_layout.addStretch()
        button_container.setLayout(button_layout)
        scroll.setWidget(button_container)
        
        area_layout = QVBoxLayout()
        area_layout.addWidget(scroll)
        area.setLayout(area_layout)
        
        return area
    
    def create_button(self, text):
        """버튼 생성"""
        button = QPushButton(text)
        button.setMinimumHeight(60)
        button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                color: white;
                background-color: #007bff;
                border: 3px solid #007bff;
                border-radius: 15px;
                padding: 15px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
                border-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
                border-color: #004085;
            }
        """)
        
        button.clicked.connect(lambda: self.on_button_click(text))
        return button
    
    def create_control_panel(self):
        """컨트롤 패널 생성"""
        panel = QWidget()
        panel.setMaximumHeight(80)
        panel.setStyleSheet("background-color: #f0f0f0; border: 2px solid #ccc; border-radius: 10px;")
        
        layout = QHBoxLayout()
        
        # 시선 추적 버튼
        self.tracking_button = QPushButton("시선 추적 시작")
        self.tracking_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #28a745;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.tracking_button.clicked.connect(self.toggle_tracking)
        layout.addWidget(self.tracking_button)
        
        # 사용자 유형 변경 버튼
        user_button = QPushButton("사용자 유형 변경")
        user_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #17a2b8;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        user_button.clicked.connect(self.change_user_type)
        layout.addWidget(user_button)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def start_gaze_tracking(self):
        """시선 추적 시작"""
        self.gaze_tracker.start_tracking()
        self.gaze_timer = QTimer()
        self.gaze_timer.timeout.connect(self.update_display)
        self.gaze_timer.start(100)  # 10 FPS
    
    def toggle_tracking(self):
        """시선 추적 토글"""
        if self.gaze_tracker.is_tracking:
            self.gaze_tracker.stop_tracking()
            self.tracking_button.setText("시선 추적 시작")
        else:
            self.gaze_tracker.start_tracking()
            self.tracking_button.setText("시선 추적 중지")
    
    def update_display(self):
        """화면 업데이트"""
        gaze = self.gaze_tracker.get_current_gaze()
        self.gaze_label.setText(f"시선 좌표: ({gaze['x']}, {gaze['y']})")
        
        # 적응형 UI 업데이트
        self.update_adaptive_ui(gaze)
    
    def update_adaptive_ui(self, gaze):
        """적응형 UI 업데이트"""
        if not gaze or gaze['confidence'] < 0.5:
            self.adaptive_label.setText("적응형 UI: 신호 없음")
            return
        
        gaze_x, gaze_y = gaze['x'], gaze['y']
        adaptive_active = False
        
        # 각 버튼에 대해 시선이 머무는지 확인
        for button_text, button in self.buttons.items():
            button_rect = button.geometry()
            button_center_x = button_rect.x() + button_rect.width() // 2
            button_center_y = button_rect.y() + button_rect.height() // 2
            
            # 시선과 버튼 중심점 간의 거리 계산
            distance = ((gaze_x - button_center_x) ** 2 + (gaze_y - button_center_y) ** 2) ** 0.5
            
            # 버튼 반경 (사용자 유형에 따라 조정)
            if self.current_user_type == "elderly":
                detection_radius = 100  # 노인은 더 넓은 감지 영역
            else:
                detection_radius = 60   # 젊은 그룹은 정확한 감지
            
            if distance <= detection_radius:
                # 버튼 확대 및 강조
                self.enhance_button(button)
                adaptive_active = True
            else:
                # 원래 상태로 복원
                self.restore_button(button)
        
        # 적응형 UI 상태 업데이트
        if adaptive_active:
            self.adaptive_label.setText("적응형 UI: 활성")
        else:
            self.adaptive_label.setText("적응형 UI: 대기")
    
    def enhance_button(self, button):
        """버튼 강화"""
        # 크기 변경
        current_height = button.height()
        new_height = int(current_height * 1.3)
        button.setFixedHeight(new_height)
        
        # 스타일 변경
        button.setStyleSheet("""
            QPushButton {
                font-size: 24px;
                font-weight: bold;
                color: white;
                background-color: #ff6b6b;
                border: 4px solid #ff4757;
                border-radius: 20px;
                padding: 20px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #ff5252;
                border-color: #ff3742;
            }
            QPushButton:pressed {
                background-color: #ff4757;
                border-color: #ff3742;
            }
        """)
        
        # 강제 업데이트
        button.update()
        button.repaint()
    
    def restore_button(self, button):
        """버튼 원래 상태로 복원"""
        # 크기 복원
        button.setFixedHeight(60)
        
        # 스타일 복원
        button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                color: white;
                background-color: #007bff;
                border: 3px solid #007bff;
                border-radius: 15px;
                padding: 15px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3;
                border-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
                border-color: #004085;
            }
        """)
        
        # 강제 업데이트
        button.update()
        button.repaint()
    
    def change_user_type(self):
        """사용자 유형 변경"""
        user_types = ["young", "elderly"]
        current_index = user_types.index(self.current_user_type) if self.current_user_type in user_types else 0
        next_index = (current_index + 1) % len(user_types)
        self.current_user_type = user_types[next_index]
        
        # 라벨 업데이트
        user_type_korean = {"young": "젊은 그룹", "elderly": "노인 그룹"}
        self.user_type_label.setText(f"사용자 유형: {user_type_korean.get(self.current_user_type, self.current_user_type)}")
        
        QMessageBox.information(self, "사용자 유형 변경", 
                              f"사용자 유형이 {user_type_korean.get(self.current_user_type, self.current_user_type)}로 변경되었습니다.")
    
    def on_button_click(self, button_text):
        """버튼 클릭 이벤트"""
        QMessageBox.information(self, "선택", f"'{button_text}'을(를) 선택했습니다!")
        
        # 시선 데이터 저장
        gaze_data = {
            'timestamp': time.time(),
            'button_clicked': button_text,
            'user_type': self.current_user_type,
            'gaze_history': self.gaze_tracker.gaze_history[-10:]
        }
        
        # 데이터를 JSON 파일로 저장
        filename = f"simple_gaze_data_{int(time.time())}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(gaze_data, f, ensure_ascii=False, indent=2)
        
        print(f"시선 데이터가 {filename}에 저장되었습니다.")

def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    app.setApplicationName("간단한 적응형 키오스크")
    app.setApplicationVersion("1.0")
    
    # 메인 윈도우 생성
    window = SimpleAdaptiveUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 