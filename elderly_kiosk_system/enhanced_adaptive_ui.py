#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 적응형 키오스크 UI 시스템
비교 분석 결과를 바탕으로 한 정교한 사용자 분류 및 UI 적응
"""

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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Beam Eye Tracker SDK import (실제 SDK 경로에 맞게 수정 필요)
try:
    import eyeware
except ImportError:
    print("Beam Eye Tracker SDK를 찾을 수 없습니다. 시뮬레이션 모드로 실행됩니다.")
    eyeware = None

class EnhancedGazeAnalyzer:
    """향상된 시선 분석기 - 비교 분석 결과 기반"""
    
    def __init__(self):
        self.gaze_history = []
        self.fixation_history = []
        self.movement_patterns = []
        self.efficiency_metrics = {}
        self.user_classification = "unknown"
        self.confidence_score = 0.0
        
        # 분석 결과 기반 임계값들
        self.elderly_thresholds = {
            'avg_distance': 150.0,  # 노인 그룹 평균 시선 이동 거리
            'avg_velocity': 200.0,  # 노인 그룹 평균 시선 속도
            'fixation_count': 8.0,   # 노인 그룹 평균 고정점 수
            'efficiency': 0.85       # 노인 그룹 평균 효율성
        }
        
        self.young_thresholds = {
            'avg_distance': 100.0,   # 젊은 그룹 평균 시선 이동 거리
            'avg_velocity': 150.0,   # 젊은 그룹 평균 시선 속도
            'fixation_count': 5.0,   # 젊은 그룹 평균 고정점 수
            'efficiency': 0.95       # 젊은 그룹 평균 효율성
        }
    
    def add_gaze_point(self, gaze_data: Dict):
        """시선 점 추가 및 분석"""
        self.gaze_history.append(gaze_data)
        
        # 최근 100개 시선 점만 유지
        if len(self.gaze_history) > 100:
            self.gaze_history = self.gaze_history[-100:]
        
        # 실시간 분석 수행
        self.analyze_current_patterns()
    
    def analyze_current_patterns(self):
        """현재 시선 패턴 분석"""
        if len(self.gaze_history) < 10:
            return
        
        # 시선 이동 거리 및 속도 계산
        distances = []
        velocities = []
        
        for i in range(1, len(self.gaze_history)):
            prev = self.gaze_history[i-1]
            curr = self.gaze_history[i]
            
            if prev['confidence'] > 0.5 and curr['confidence'] > 0.5:
                dx = curr['x'] - prev['x']
                dy = curr['y'] - prev['y']
                distance = np.sqrt(dx**2 + dy**2)
                distances.append(distance)
                
                dt = curr['timestamp'] - prev['timestamp']
                if dt > 0:
                    velocity = distance / dt
                    velocities.append(velocity)
        
        # 고정점 분석
        fixations = self.detect_fixations()
        
        # 효율성 계산
        valid_points = len([p for p in self.gaze_history if p['confidence'] > 0.5])
        efficiency = valid_points / len(self.gaze_history) if self.gaze_history else 0
        
        # 현재 메트릭 저장
        current_metrics = {
            'avg_distance': np.mean(distances) if distances else 0,
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'fixation_count': len(fixations),
            'efficiency': efficiency,
            'timestamp': time.time()
        }
        
        self.movement_patterns.append(current_metrics)
        
        # 사용자 분류 업데이트
        self.classify_user(current_metrics)
    
    def detect_fixations(self, threshold: float = 30.0, min_duration: float = 0.3) -> List[Dict]:
        """고정점 감지"""
        fixations = []
        current_fixation = None
        
        for gaze in self.gaze_history:
            if gaze['confidence'] < 0.5:
                continue
            
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
                
                if distance <= threshold:
                    current_fixation['points'].append(gaze)
                else:
                    # 고정점 종료 및 저장
                    duration = gaze['timestamp'] - current_fixation['start_time']
                    if duration >= min_duration:
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
        
        return fixations
    
    def classify_user(self, metrics: Dict):
        """사용자 유형 분류 (비교 분석 결과 기반)"""
        # 노인 그룹 특성과의 유사도 계산
        elderly_similarity = 0
        young_similarity = 0
        
        # 시선 이동 거리 비교
        if metrics['avg_distance'] > self.elderly_thresholds['avg_distance']:
            elderly_similarity += 1
        elif metrics['avg_distance'] < self.young_thresholds['avg_distance']:
            young_similarity += 1
        
        # 시선 속도 비교
        if metrics['avg_velocity'] > self.elderly_thresholds['avg_velocity']:
            elderly_similarity += 1
        elif metrics['avg_velocity'] < self.young_thresholds['avg_velocity']:
            young_similarity += 1
        
        # 고정점 수 비교
        if metrics['fixation_count'] > self.elderly_thresholds['fixation_count']:
            elderly_similarity += 1
        elif metrics['fixation_count'] < self.young_thresholds['fixation_count']:
            young_similarity += 1
        
        # 효율성 비교
        if metrics['efficiency'] < self.elderly_thresholds['efficiency']:
            elderly_similarity += 1
        elif metrics['efficiency'] > self.young_thresholds['efficiency']:
            young_similarity += 1
        
        # 분류 결정
        total_features = 4
        elderly_score = elderly_similarity / total_features
        young_score = young_similarity / total_features
        
        if elderly_score > 0.6:
            self.user_classification = "elderly"
            self.confidence_score = elderly_score
        elif young_score > 0.6:
            self.user_classification = "young"
            self.confidence_score = young_score
        else:
            self.user_classification = "middle"
            self.confidence_score = max(elderly_score, young_score)
    
    def get_user_classification(self) -> Tuple[str, float]:
        """사용자 분류 결과 반환"""
        return self.user_classification, self.confidence_score
    
    def get_analysis_summary(self) -> Dict:
        """분석 요약 반환"""
        if not self.movement_patterns:
            return {}
        
        latest = self.movement_patterns[-1]
        return {
            'user_type': self.user_classification,
            'confidence': self.confidence_score,
            'avg_distance': latest['avg_distance'],
            'avg_velocity': latest['avg_velocity'],
            'fixation_count': latest['fixation_count'],
            'efficiency': latest['efficiency'],
            'total_gaze_points': len(self.gaze_history)
        }

class EnhancedGazeTracker:
    """향상된 시선 추적기"""
    
    def __init__(self, simulation_mode: bool = True):
        self.simulation_mode = simulation_mode
        self.gaze_analyzer = EnhancedGazeAnalyzer()
        self.is_tracking = False
        self.current_gaze = {'x': 0, 'y': 0, 'confidence': 0, 'timestamp': time.time()}
        
        if not simulation_mode and eyeware:
            self.init_beam_tracker()
        else:
            self.init_simulation()
    
    def init_beam_tracker(self):
        """Beam Eye Tracker 초기화"""
        try:
            # Beam Eye Tracker SDK 초기화 코드
            pass
        except Exception as e:
            print(f"Beam Eye Tracker 초기화 실패: {e}")
            self.simulation_mode = True
            self.init_simulation()
    
    def init_simulation(self):
        """시뮬레이션 모드 초기화"""
        print("향상된 시선 추적 시뮬레이션 모드 활성화")
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
        """향상된 시선 추적 시뮬레이션"""
        import random
        
        # 사용자 유형별 다른 시선 패턴 시뮬레이션
        user_types = ["young", "middle", "elderly"]
        current_user_type = random.choice(user_types)
        
        while self.is_tracking:
            # 사용자 유형에 따른 시선 패턴 생성
            if current_user_type == "elderly":
                # 노인 그룹: 더 큰 이동 거리, 더 많은 고정점
                x = random.randint(300, 1400)
                y = random.randint(200, 900)
                confidence = random.uniform(0.6, 0.9)  # 낮은 효율성
            elif current_user_type == "young":
                # 젊은 그룹: 작은 이동 거리, 높은 효율성
                x = random.randint(500, 1100)
                y = random.randint(300, 700)
                confidence = random.uniform(0.8, 1.0)  # 높은 효율성
            else:
                # 중간 그룹: 중간 수준
                x = random.randint(400, 1200)
                y = random.randint(250, 800)
                confidence = random.uniform(0.7, 0.95)
            
            self.current_gaze = {
                'x': x,
                'y': y,
                'confidence': confidence,
                'timestamp': time.time()
            }
            
            # 분석기에 시선 데이터 추가
            self.gaze_analyzer.add_gaze_point(self.current_gaze)
            
            time.sleep(0.033)  # 30 FPS
    
    def get_current_gaze(self) -> Dict:
        """현재 시선 좌표 반환"""
        return self.current_gaze
    
    def get_user_classification(self) -> Tuple[str, float]:
        """사용자 분류 결과 반환"""
        return self.gaze_analyzer.get_user_classification()
    
    def get_analysis_summary(self) -> Dict:
        """분석 요약 반환"""
        return self.gaze_analyzer.get_analysis_summary()

class EnhancedAdaptiveUIWidget(QWidget):
    """향상된 적응형 UI 위젯"""
    
    def __init__(self):
        super().__init__()
        self.gaze_tracker = EnhancedGazeTracker(simulation_mode=True)
        self.current_user_type = "unknown"
        self.ui_settings = self.get_ui_settings("elderly")  # 기본값
        self.adaptive_buttons = {}
        self.fixation_threshold = 1.0
        self.zoom_factor = 1.3
        self.adaptation_level = 0  # 0: 기본, 1: 약간, 2: 강화
        
        # 버튼 업데이트를 위한 타이머
        self.button_update_timer = QTimer()
        self.button_update_timer.timeout.connect(self.force_layout_update)
        self.button_update_timer.setSingleShot(True)
        
        self.init_ui()
        self.start_gaze_tracking()
    
    def get_ui_settings(self, user_type: str) -> Dict:
        """사용자 유형에 따른 UI 설정 (분석 결과 기반)"""
        settings = {
            'young': {
                'font_size': 16,
                'button_height': 40,
                'button_width': 120,
                'text_color': '#333333',
                'background_color': '#ffffff',
                'button_color': '#007bff',
                'button_hover_color': '#0056b3',
                'spacing': 10,
                'animation_speed': 0.2
            },
            'middle': {
                'font_size': 20,
                'button_height': 55,
                'button_width': 150,
                'text_color': '#222222',
                'background_color': '#f8f9fa',
                'button_color': '#28a745',
                'button_hover_color': '#1e7e34',
                'spacing': 15,
                'animation_speed': 0.3
            },
            'elderly': {
                'font_size': 28,
                'button_height': 80,
                'button_width': 200,
                'text_color': '#000000',
                'background_color': '#ffffff',
                'button_color': '#dc3545',
                'button_hover_color': '#c82333',
                'spacing': 25,
                'animation_speed': 0.5
            }
        }
        return settings.get(user_type, settings['elderly'])
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("향상된 적응형 키오스크 UI")
        self.setGeometry(100, 100, 1400, 900)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        
        # 상단 정보 패널
        info_panel = self.create_enhanced_info_panel()
        main_layout.addWidget(info_panel)
        
        # 메뉴 버튼들
        self.menu_layout = QGridLayout()
        menu_items = [
            "아메리카노", "카페라떼", "카푸치노", "에스프레소",
            "수박주스", "오렌지주스", "사과주스", "포도주스",
            "햄&치즈 샌드위치", "치킨 샌드위치", "베지 샌드위치", "터키 샌드위치"
        ]
        
        row, col = 0, 0
        for item in menu_items:
            button = self.create_enhanced_button(item)
            self.menu_layout.addWidget(button, row, col)
            self.adaptive_buttons[item] = button
            col += 1
            if col >= 4:
                col = 0
                row += 1
        
        menu_widget = QWidget()
        menu_widget.setLayout(self.menu_layout)
        main_layout.addWidget(menu_widget)
        
        # 하단 컨트롤 패널
        control_panel = self.create_enhanced_control_panel()
        main_layout.addWidget(control_panel)
        
        self.setLayout(main_layout)
        self.apply_ui_settings()
    
    def create_enhanced_info_panel(self) -> QWidget:
        """향상된 정보 패널 생성"""
        panel = QWidget()
        panel.setMaximumHeight(120)
        panel.setStyleSheet("background-color: #f8f9fa; border: 2px solid #ddd; border-radius: 10px;")
        
        layout = QHBoxLayout()
        
        # 사용자 분류 정보
        self.user_classification_label = QLabel("사용자 분류: 분석 중...")
        self.user_classification_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        layout.addWidget(self.user_classification_label)
        
        # 신뢰도 점수
        self.confidence_label = QLabel("신뢰도: 0%")
        self.confidence_label.setStyleSheet("font-size: 14px; color: #666;")
        layout.addWidget(self.confidence_label)
        
        # 시선 추적 상태
        self.gaze_status_label = QLabel("시선 추적: 비활성")
        self.gaze_status_label.setStyleSheet("font-size: 14px; color: #666;")
        layout.addWidget(self.gaze_status_label)
        
        # 적응형 UI 상태
        self.adaptive_status_label = QLabel("적응형 UI: 대기")
        self.adaptive_status_label.setStyleSheet("font-size: 14px; color: #666;")
        layout.addWidget(self.adaptive_status_label)
        
        # 현재 시선 좌표
        self.gaze_coord_label = QLabel("시선 좌표: (0, 0)")
        self.gaze_coord_label.setStyleSheet("font-size: 14px; color: #666;")
        layout.addWidget(self.gaze_coord_label)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_enhanced_button(self, text: str) -> QPushButton:
        """향상된 적응형 버튼 생성"""
        button = QPushButton(text)
        # 고정 크기로 설정
        button.setFixedSize(self.ui_settings['button_width'], self.ui_settings['button_height'])
        button.setStyleSheet(f"""
            QPushButton {{
                font-size: {self.ui_settings['font_size']}px;
                font-weight: bold;
                color: white;
                background-color: {self.ui_settings['button_color']};
                border: 3px solid {self.ui_settings['button_color']};
                border-radius: 15px;
                padding: 15px;
            }}
            QPushButton:hover {{
                background-color: {self.ui_settings['button_hover_color']};
                border-color: {self.ui_settings['button_hover_color']};
            }}
            QPushButton:pressed {{
                background-color: #666666;
            }}
        """)
        
        button.clicked.connect(lambda: self.on_button_click(text))
        return button
    
    def create_enhanced_control_panel(self) -> QWidget:
        """향상된 컨트롤 패널 생성"""
        panel = QWidget()
        panel.setMaximumHeight(100)
        panel.setStyleSheet("background-color: #f8f9fa; border: 2px solid #ddd; border-radius: 10px;")
        
        layout = QHBoxLayout()
        
        # 시선 추적 시작/중지 버튼
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
        self.tracking_button.clicked.connect(self.toggle_gaze_tracking)
        layout.addWidget(self.tracking_button)
        
        # 분석 결과 표시 버튼
        analysis_button = QPushButton("상세 분석")
        analysis_button.setStyleSheet("""
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
        analysis_button.clicked.connect(self.show_detailed_analysis)
        layout.addWidget(analysis_button)
        
        # 수동 사용자 유형 변경 버튼
        manual_button = QPushButton("수동 설정")
        manual_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #ffc107;
                color: black;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
        """)
        manual_button.clicked.connect(self.manual_user_setting)
        layout.addWidget(manual_button)
        
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
        self.gaze_timer.timeout.connect(self.update_enhanced_display)
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
    
    def update_enhanced_display(self):
        """향상된 표시 업데이트"""
        gaze = self.gaze_tracker.get_current_gaze()
        self.gaze_coord_label.setText(f"시선 좌표: ({gaze['x']}, {gaze['y']})")
        
        # 사용자 분류 업데이트
        user_type, confidence = self.gaze_tracker.get_user_classification()
        if user_type != self.current_user_type:
            self.current_user_type = user_type
            self.ui_settings = self.get_ui_settings(user_type)
            self.apply_ui_settings()
            self.update_all_buttons()
        
        # 라벨 업데이트
        user_type_korean = {"young": "젊은 그룹", "middle": "중간 그룹", "elderly": "노인 그룹", "unknown": "분석 중"}
        self.user_classification_label.setText(f"사용자 분류: {user_type_korean.get(user_type, user_type)}")
        self.confidence_label.setText(f"신뢰도: {confidence:.1%}")
        
        # 적응형 UI 업데이트
        self.update_enhanced_adaptive_ui(gaze)
    
    def update_enhanced_adaptive_ui(self, current_gaze: Dict):
        """향상된 적응형 UI 업데이트"""
        if not current_gaze or current_gaze['confidence'] < 0.5:
            self.adaptive_status_label.setText("적응형 UI: 신호 없음")
            return
        
        gaze_x, gaze_y = current_gaze['x'], current_gaze['y']
        adaptive_active = False
        
        # 각 버튼에 대해 시선이 머무는지 확인
        for button_text, button in self.adaptive_buttons.items():
            button_rect = button.geometry()
            button_center_x = button_rect.x() + button_rect.width() // 2
            button_center_y = button_rect.y() + button_rect.height() // 2
            
            # 시선과 버튼 중심점 간의 거리 계산
            distance = ((gaze_x - button_center_x) ** 2 + (gaze_y - button_center_y) ** 2) ** 0.5
            
            # 버튼 반경 (사용자 유형에 따라 조정)
            base_radius = max(button_rect.width(), button_rect.height()) // 2
            if self.current_user_type == "elderly":
                detection_radius = base_radius * 2.0  # 노인은 더 넓은 감지 영역
            elif self.current_user_type == "young":
                detection_radius = base_radius * 1.2  # 젊은 그룹은 정확한 감지
            else:
                detection_radius = base_radius * 1.5  # 중간 그룹
            
            if distance <= detection_radius:
                # 고정 시간 확인
                fixation_duration = self.check_enhanced_fixation_duration(gaze_x, gaze_y)
                
                if fixation_duration >= self.fixation_threshold:
                    # 강화된 적응 (노인 그룹)
                    if self.current_user_type == "elderly":
                        self.enhance_button_for_elderly(button)
                    else:
                        self.enhance_button(button)
                    adaptive_active = True
                else:
                    # 약간의 강조
                    self.highlight_button(button)
                    adaptive_active = True
            else:
                # 원래 상태로 복원
                self.restore_button(button)
        
        # 적응형 UI 상태 업데이트
        if adaptive_active:
            self.adaptive_status_label.setText("적응형 UI: 활성")
        else:
            self.adaptive_status_label.setText("적응형 UI: 대기")
    
    def check_enhanced_fixation_duration(self, gaze_x: int, gaze_y: int) -> float:
        """향상된 고정 시간 계산"""
        # 사용자 유형에 따른 감지 반경 조정
        if self.current_user_type == "elderly":
            detection_radius = 80  # 노인은 더 넓은 감지 영역
        elif self.current_user_type == "young":
            detection_radius = 40  # 젊은 그룹은 정확한 감지
        else:
            detection_radius = 60  # 중간 그룹
        
        # 시선 히스토리에서 고정 시간 계산
        gaze_history = self.gaze_tracker.gaze_analyzer.gaze_history
        if len(gaze_history) < 10:
            return 0.0
        
        recent_gaze = gaze_history[-30:]
        fixation_start = None
        total_duration = 0.0
        
        for gaze in recent_gaze:
            if gaze['confidence'] < 0.5:
                continue
            
            distance = ((gaze['x'] - gaze_x) ** 2 + (gaze['y'] - gaze_y) ** 2) ** 0.5
            
            if distance <= detection_radius:
                if fixation_start is None:
                    fixation_start = gaze['timestamp']
            else:
                if fixation_start is not None:
                    duration = gaze['timestamp'] - fixation_start
                    total_duration += duration
                    fixation_start = None
        
        return total_duration
    
    def enhance_button_for_elderly(self, button: QPushButton):
        """노인 그룹을 위한 강화된 버튼 효과"""
        new_width = int(self.ui_settings['button_width'] * 1.3)
        new_height = int(self.ui_settings['button_height'] * 1.3)
        
        # 버튼 크기 강제 변경 - 여러 방법 시도
        button.setFixedSize(new_width, new_height)
        button.resize(new_width, new_height)
        button.setMinimumSize(new_width, new_height)
        button.setMaximumSize(new_width, new_height)
        
        # 부모 위젯에 크기 변경 알림
        button.parent().updateGeometry()
        
        button.setStyleSheet(f"""
            QPushButton {{
                font-size: {int(self.ui_settings['font_size'] * 1.4)}px;
                font-weight: bold;
                color: white;
                background-color: #ff6b6b;
                border: 4px solid #ff4757;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 6px 12px rgba(0,0,0,0.4);
            }}
            QPushButton:hover {{
                background-color: #ff5252;
                border-color: #ff3742;
            }}
        """)
        
        # 강제 업데이트
        button.update()
        button.repaint()
        
        # 레이아웃 강제 업데이트
        if hasattr(self, 'menu_layout'):
            self.menu_layout.update()
            self.menu_layout.activate()
            # 지연된 강제 업데이트
            self.button_update_timer.start(100)
    
    def force_layout_update(self):
        """레이아웃 강제 업데이트"""
        if hasattr(self, 'menu_layout'):
            self.menu_layout.update()
            self.menu_layout.activate()
            self.updateGeometry()
            self.repaint()
    
    def enhance_button(self, button: QPushButton):
        """일반 강화된 버튼 효과"""
        new_width = int(self.ui_settings['button_width'] * 1.2)
        new_height = int(self.ui_settings['button_height'] * 1.2)
        
        # 버튼 크기 강제 변경 - 여러 방법 시도
        button.setFixedSize(new_width, new_height)
        button.resize(new_width, new_height)
        button.setMinimumSize(new_width, new_height)
        button.setMaximumSize(new_width, new_height)
        
        # 부모 위젯에 크기 변경 알림
        button.parent().updateGeometry()
        
        button.setStyleSheet(f"""
            QPushButton {{
                font-size: {int(self.ui_settings['font_size'] * 1.2)}px;
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
        
        # 강제 업데이트
        button.update()
        button.repaint()
        
        # 레이아웃 강제 업데이트
        if hasattr(self, 'menu_layout'):
            self.menu_layout.update()
            self.menu_layout.activate()
            # 지연된 강제 업데이트
            self.button_update_timer.start(100)
    
    def highlight_button(self, button: QPushButton):
        """버튼 강조 효과"""
        button.setStyleSheet(f"""
            QPushButton {{
                font-size: {self.ui_settings['font_size']}px;
                font-weight: bold;
                color: white;
                background-color: #ffc107;
                border: 3px solid #ff9800;
                border-radius: 15px;
                padding: 15px;
            }}
            QPushButton:hover {{
                background-color: #ffb300;
                border-color: #ff8f00;
            }}
        """)
    
    def restore_button(self, button: QPushButton):
        """버튼 원래 상태로 복원"""
        # 원래 크기로 복원 - 여러 방법 시도
        button.setFixedSize(self.ui_settings['button_width'], self.ui_settings['button_height'])
        button.resize(self.ui_settings['button_width'], self.ui_settings['button_height'])
        button.setMinimumSize(self.ui_settings['button_width'], self.ui_settings['button_height'])
        button.setMaximumSize(self.ui_settings['button_width'], self.ui_settings['button_height'])
        
        # 부모 위젯에 크기 변경 알림
        button.parent().updateGeometry()
        
        button.setStyleSheet(f"""
            QPushButton {{
                font-size: {self.ui_settings['font_size']}px;
                font-weight: bold;
                color: white;
                background-color: {self.ui_settings['button_color']};
                border: 3px solid {self.ui_settings['button_color']};
                border-radius: 15px;
                padding: 15px;
            }}
            QPushButton:hover {{
                background-color: {self.ui_settings['button_hover_color']};
                border-color: {self.ui_settings['button_hover_color']};
            }}
        """)
        
        # 강제 업데이트
        button.update()
        button.repaint()
        
        # 레이아웃 강제 업데이트
        if hasattr(self, 'menu_layout'):
            self.menu_layout.update()
            self.menu_layout.activate()
            # 지연된 강제 업데이트
            self.button_update_timer.start(100)
    
    def update_all_buttons(self):
        """모든 버튼 스타일 업데이트"""
        for button in self.adaptive_buttons.values():
            # 크기 업데이트
            button.setFixedSize(self.ui_settings['button_width'], self.ui_settings['button_height'])
            
            button.setStyleSheet(f"""
                QPushButton {{
                    font-size: {self.ui_settings['font_size']}px;
                    font-weight: bold;
                    color: white;
                    background-color: {self.ui_settings['button_color']};
                    border: 3px solid {self.ui_settings['button_color']};
                    border-radius: 15px;
                    padding: 15px;
                }}
                QPushButton:hover {{
                    background-color: {self.ui_settings['button_hover_color']};
                    border-color: {self.ui_settings['button_hover_color']};
                }}
            """)
    
    def on_button_click(self, button_text: str):
        """버튼 클릭 이벤트"""
        QMessageBox.information(self, "선택", f"'{button_text}'을(를) 선택했습니다!")
        
        # 시선 데이터 저장
        analysis_summary = self.gaze_tracker.get_analysis_summary()
        click_data = {
            'timestamp': time.time(),
            'button_clicked': button_text,
            'user_classification': analysis_summary.get('user_type', 'unknown'),
            'confidence': analysis_summary.get('confidence', 0.0),
            'analysis_metrics': analysis_summary
        }
        
        # 데이터를 JSON 파일로 저장
        filename = f"enhanced_gaze_data_{int(time.time())}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(click_data, f, ensure_ascii=False, indent=2)
        
        print(f"향상된 시선 데이터가 {filename}에 저장되었습니다.")
    
    def show_detailed_analysis(self):
        """상세 분석 결과 표시"""
        analysis_summary = self.gaze_tracker.get_analysis_summary()
        
        if not analysis_summary:
            QMessageBox.warning(self, "분석 불가", "분석 데이터가 충분하지 않습니다.")
            return
        
        analysis_text = f"""
상세 분석 결과:

📊 사용자 분류 정보:
- 분류된 유형: {analysis_summary.get('user_type', 'unknown')}
- 신뢰도: {analysis_summary.get('confidence', 0):.1%}
- 총 시선 점 수: {analysis_summary.get('total_gaze_points', 0)}

📈 시선 패턴 분석:
- 평균 시선 이동 거리: {analysis_summary.get('avg_distance', 0):.1f} 픽셀
- 평균 시선 속도: {analysis_summary.get('avg_velocity', 0):.1f} 픽셀/초
- 고정점 수: {analysis_summary.get('fixation_count', 0)}
- 시선 효율성: {analysis_summary.get('efficiency', 0):.1%}

🎯 UI 적응 수준:
- 현재 UI 설정: {self.current_user_type}
- 폰트 크기: {self.ui_settings['font_size']}px
- 버튼 크기: {self.ui_settings['button_width']}x{self.ui_settings['button_height']}px
        """
        
        QMessageBox.information(self, "상세 분석", analysis_text)
    
    def manual_user_setting(self):
        """수동 사용자 유형 설정"""
        user_types = ["young", "middle", "elderly"]
        current_index = user_types.index(self.current_user_type) if self.current_user_type in user_types else 0
        next_index = (current_index + 1) % len(user_types)
        self.current_user_type = user_types[next_index]
        
        # UI 설정 업데이트
        self.ui_settings = self.get_ui_settings(self.current_user_type)
        self.apply_ui_settings()
        self.update_all_buttons()
        
        user_type_korean = {"young": "젊은 그룹", "middle": "중간 그룹", "elderly": "노인 그룹"}
        QMessageBox.information(self, "수동 설정", f"UI 설정이 {user_type_korean.get(self.current_user_type, self.current_user_type)} 모드로 변경되었습니다.")

class EnhancedAdaptiveKioskApp(QApplication):
    """향상된 적응형 키오스크 애플리케이션"""
    
    def __init__(self, argv):
        super().__init__(argv)
        self.setApplicationName("향상된 적응형 키오스크")
        self.setApplicationVersion("2.0")
        
        # 메인 윈도우 생성
        self.main_window = EnhancedAdaptiveUIWidget()
        self.main_window.show()

def main():
    """메인 함수"""
    app = EnhancedAdaptiveKioskApp(sys.argv)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 