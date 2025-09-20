#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
노인 친화적 키오스크 시스템 실행 스크립트
"""

import sys
import os
import subprocess
import argparse

def check_dependencies():
    """필요한 패키지가 설치되어 있는지 확인"""
    required_packages = [
        'PyQt5', 'numpy', 'pandas', 'sklearn', 
        'matplotlib', 'seaborn', 'cv2'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'cv2':
                import cv2
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 다음 패키지가 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n다음 명령어로 설치하세요:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ 모든 필요한 패키지가 설치되어 있습니다.")
    return True

def run_data_analysis():
    """데이터 분석 실행"""
    print("📊 데이터 분석을 시작합니다...")
    try:
        from elderly_ui_analyzer import ElderlyUIAnalyzer
        
        analyzer = ElderlyUIAnalyzer()
        analyzer.load_all_user_data()
        analyzer.analyze_gaze_patterns()
        analyzer.train_ml_model()
        
        report = analyzer.generate_report()
        print("\n" + "="*50)
        print("분석 완료!")
        print("="*50)
        print(report)
        
        # 리포트를 파일로 저장
        with open("analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("\n📄 분석 리포트가 'analysis_report.txt'에 저장되었습니다.")
        
    except Exception as e:
        print(f"❌ 데이터 분석 중 오류 발생: {e}")

def run_adaptive_ui():
    """적응형 UI 시스템 실행"""
    print("🖥️ 적응형 UI 시스템을 시작합니다...")
    try:
        from adaptive_ui_system import AdaptiveUIWidget
        from PyQt5.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        ui = AdaptiveUIWidget(user_type="elderly")
        ui.show()
        
        print("✅ UI 시스템이 시작되었습니다. 창을 닫으면 종료됩니다.")
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"❌ UI 시스템 실행 중 오류 발생: {e}")

def run_integrated_system():
    """통합 시스템 실행"""
    print("🚀 통합 키오스크 시스템을 시작합니다...")
    try:
        from elderly_kiosk_system import main
        main()
        
    except Exception as e:
        print(f"❌ 통합 시스템 실행 중 오류 발생: {e}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="노인 친화적 키오스크 시스템")
    parser.add_argument(
        'mode', 
        choices=['analyze', 'ui', 'integrated', 'all'],
        help='실행 모드 선택'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("👴 노인 친화적 키오스크 시스템")
    print("=" * 60)
    
    # 의존성 확인
    if not check_dependencies():
        return
    
    print()
    
    if args.mode == 'analyze':
        run_data_analysis()
    elif args.mode == 'ui':
        run_adaptive_ui()
    elif args.mode == 'integrated':
        run_integrated_system()
    elif args.mode == 'all':
        print("🔄 전체 시스템을 순차적으로 실행합니다...")
        print("\n1️⃣ 데이터 분석 실행...")
        run_data_analysis()
        print("\n2️⃣ 적응형 UI 시스템 실행...")
        run_adaptive_ui()

if __name__ == "__main__":
    main() 