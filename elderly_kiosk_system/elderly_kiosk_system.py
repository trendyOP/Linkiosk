#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
노인 친화적 키오스크 시스템
시선 추적 기반 적응형 UI + 머신러닝 분석
"""

import sys
import os
import json
import time
import threading
import numpy as np
from typing import Dict, List, Optional
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage

# 로컬 모듈 import
from elderly_ui_analyzer import ElderlyUIAnalyzer
from adaptive_ui_system import AdaptiveUIWidget, GazeTracker

class ElderlyKioskSystem:
    """노인 친화적 키오스크 시스템 메인 클래스"""
    
    def __init__(self):
        self.analyzer = ElderlyUIAnalyzer()
        self.gaze_tracker = GazeTracker(simulation_mode=True)
        self.current_user_type = "elderly"
        self.ui_widget = None
        self.analysis_results = {}
        
        # 시스템 초기화
        self.initialize_system()
    
    def initialize_system(self):
        """시스템 초기화"""
        print("노인 친화적 키오스크 시스템 초기화 중...")
        
        # 기존 데이터 분석
        try:
            self.analyzer.load_all_user_data()
            self.analyzer.analyze_gaze_patterns()
            self.analyzer.train_ml_model()
            print("기존 데이터 분석 완료")
        except Exception as e:
            print(f"기존 데이터 분석 실패: {e}")
        
        # 시선 추적기 초기화
        self.gaze_tracker.start_tracking()
        print("시선 추적기 초기화 완료")
    
    def start_adaptive_ui(self):
        """적응형 UI 시작"""
        app = QApplication(sys.argv)
        
        # 메인 윈도우 생성
        self.ui_widget = AdaptiveUIWidget(user_type=self.current_user_type)
        self.ui_widget.show()
        
        # 실시간 분석 타이머 설정
        analysis_timer = QTimer()
        analysis_timer.timeout.connect(self.real_time_analysis)
        analysis_timer.start(5000)  # 5초마다 분석
        
        sys.exit(app.exec_())
    
    def real_time_analysis(self):
        """실시간 시선 분석"""
        if not self.ui_widget:
            return
        
        # 현재 시선 데이터 수집
        current_gaze = self.gaze_tracker.get_current_gaze()
        gaze_history = self.gaze_tracker.get_gaze_history()
        
        if len(gaze_history) < 10:
            return
        
        # 특성 추출
        recent_gaze = gaze_history[-30:]  # 최근 30개 시선 점
        
        # 시선 이동 거리 계산
        distances = []
        velocities = []
        
        for i in range(1, len(recent_gaze)):
            if recent_gaze[i]['confidence'] > 0 and recent_gaze[i-1]['confidence'] > 0:
                dx = recent_gaze[i]['x'] - recent_gaze[i-1]['x']
                dy = recent_gaze[i]['y'] - recent_gaze[i-1]['y']
                distance = (dx**2 + dy**2)**0.5
                distances.append(distance)
                
                dt = recent_gaze[i]['timestamp'] - recent_gaze[i-1]['timestamp']
                if dt > 0:
                    velocity = distance / dt
                    velocities.append(velocity)
        
        # 고정점 분석
        fixations = self._analyze_fixations(recent_gaze)
        
        # 특성 벡터 생성
        features = [
            np.mean(distances) if distances else 0,
            np.mean(velocities) if velocities else 0,
            np.max(velocities) if velocities else 0,
            len(fixations),
            np.mean([f['duration'] for f in fixations]) if fixations else 0,
            50  # 예시 나이 (실제로는 사용자 입력 또는 추정)
        ]
        
        # 사용자 유형 예측
        try:
            prediction = self.analyzer.predict_user_type(features)
            new_user_type = prediction['user_type']
            confidence = prediction['confidence']
            
            # 신뢰도가 높고 사용자 유형이 변경된 경우 UI 업데이트
            if confidence > 0.7 and new_user_type != self.current_user_type:
                self.update_ui_for_user_type(new_user_type)
                print(f"사용자 유형 변경: {self.current_user_type} -> {new_user_type} (신뢰도: {confidence:.2f})")
                
        except Exception as e:
            print(f"실시간 분석 오류: {e}")
    
    def _analyze_fixations(self, gaze_data: List[Dict]) -> List[Dict]:
        """고정점 분석"""
        fixations = []
        current_fixation = None
        
        for gaze in gaze_data:
            if gaze['confidence'] == 0:
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
                
                distance = ((gaze['x'] - center_x)**2 + (gaze['y'] - center_y)**2)**0.5
                
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
        
        return fixations
    
    def update_ui_for_user_type(self, user_type: str):
        """사용자 유형에 따른 UI 업데이트"""
        self.current_user_type = user_type
        
        if self.ui_widget:
            # UI 설정 변경
            self.ui_widget.user_type = user_type
            self.ui_widget.ui_settings = self.ui_widget.get_ui_settings(user_type)
            self.ui_widget.apply_ui_settings()
            self.ui_widget.update_all_buttons()
            
            # 윈도우 제목 업데이트
            self.ui_widget.setWindowTitle(f"적응형 키오스크 UI ({user_type})")
            
            # 정보 패널 업데이트
            for child in self.ui_widget.findChildren(QLabel):
                if "사용자 유형:" in child.text():
                    child.setText(f"사용자 유형: {user_type.upper()}")
                    break
    
    def generate_comprehensive_report(self) -> str:
        """종합 분석 리포트 생성"""
        report = []
        report.append("=" * 60)
        report.append("노인 친화적 키오스크 시스템 종합 분석 리포트")
        report.append("=" * 60)
        report.append("")
        
        # 기존 데이터 분석 결과
        if self.analyzer.analysis_results:
            report.append("1. 기존 사용자 데이터 분석 결과")
            report.append("-" * 40)
            
            total_users = len(self.analyzer.analysis_results)
            ages = [analysis['age'] for analysis in self.analyzer.analysis_results.values()]
            avg_age = sum(ages) / len(ages)
            
            report.append(f"총 분석 사용자 수: {total_users}명")
            report.append(f"평균 나이: {avg_age:.1f}세")
            report.append(f"나이 범위: {min(ages)}세 ~ {max(ages)}세")
            report.append("")
            
            # 사용자별 상세 분석
            for user_id, analysis in self.analyzer.analysis_results.items():
                stats = analysis['gaze_statistics']
                report.append(f"사용자 {user_id} ({analysis['age']}세):")
                report.append(f"  - 평균 시선 이동 거리: {stats['average_distance']:.2f} 픽셀")
                report.append(f"  - 평균 시선 속도: {stats['average_velocity']:.2f} 픽셀/초")
                report.append(f"  - 고정점 수: {stats['fixation_count']}")
                report.append(f"  - 평균 고정 시간: {stats['average_fixation_duration']:.2f}초")
                report.append("")
        
        # 현재 시스템 상태
        report.append("2. 현재 시스템 상태")
        report.append("-" * 40)
        report.append(f"현재 사용자 유형: {self.current_user_type}")
        report.append(f"시선 추적 상태: {'활성' if self.gaze_tracker.is_tracking else '비활성'}")
        report.append(f"수집된 시선 점 수: {len(self.gaze_tracker.get_gaze_history())}")
        report.append("")
        
        # UI 권장사항
        report.append("3. UI 권장사항")
        report.append("-" * 40)
        recommendations = self.analyzer.generate_ui_recommendations(self.current_user_type)
        for key, value in recommendations.items():
            report.append(f"  - {key}: {value}")
        report.append("")
        
        # 시스템 개선 제안
        report.append("4. 시스템 개선 제안")
        report.append("-" * 40)
        if self.current_user_type == "elderly":
            report.append("  - 폰트 크기: 24px 이상 권장")
            report.append("  - 버튼 크기: 70px 높이 이상 권장")
            report.append("  - 색상 대비: 높은 대비 사용")
            report.append("  - 애니메이션: 느린 속도 권장")
            report.append("  - 터치 영역: 충분히 큰 터치 영역 제공")
        elif self.current_user_type == "middle":
            report.append("  - 폰트 크기: 18px 권장")
            report.append("  - 버튼 크기: 50px 높이 권장")
            report.append("  - 색상 대비: 중간 대비 사용")
            report.append("  - 애니메이션: 중간 속도 권장")
        else:
            report.append("  - 표준 UI 설정 사용")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)

class ElderlyKioskApp(QApplication):
    """노인 친화적 키오스크 애플리케이션"""
    
    def __init__(self, argv):
        super().__init__(argv)
        self.setApplicationName("노인 친화적 키오스크")
        self.setApplicationVersion("1.0")
        
        # 시스템 초기화
        self.kiosk_system = ElderlyKioskSystem()
        
        # 메인 윈도우 생성
        self.main_window = self.kiosk_system.ui_widget = AdaptiveUIWidget(
            user_type=self.kiosk_system.current_user_type
        )
        self.main_window.show()
        
        # 실시간 분석 타이머 설정
        self.analysis_timer = QTimer()
        self.analysis_timer.timeout.connect(self.kiosk_system.real_time_analysis)
        self.analysis_timer.start(5000)  # 5초마다 분석

def main():
    """메인 함수"""
    app = ElderlyKioskApp(sys.argv)
    
    # 종료 시 리포트 생성
    def on_about_to_quit():
        report = app.kiosk_system.generate_comprehensive_report()
        with open("elderly_kiosk_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("분석 리포트가 elderly_kiosk_report.txt에 저장되었습니다.")
    
    app.aboutToQuit.connect(on_about_to_quit)
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 