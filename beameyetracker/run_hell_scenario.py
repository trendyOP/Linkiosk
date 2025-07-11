#!/usr/bin/env python3
"""
헬 난이도 키오스크 시나리오 실행 스크립트
고령자 UI 사용성 연구를 위한 복잡한 시나리오 데이터 수집
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """필요한 의존성 확인"""
    print("🔍 의존성 확인 중...")
    
    # Python 패키지 확인
    required_packages = [
        'numpy', 'matplotlib', 'requests', 'keyboard'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} (설치 필요)")
    
    if missing_packages:
        print(f"\n📦 다음 패키지를 설치해주세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    # Eyeware SDK 확인
    try:
        from eyeware import beam_eye_tracker as bet
        print("✅ Eyeware SDK")
    except ImportError:
        print("⚠️  Eyeware SDK (시선 추적 비활성화)")
    
    # kiosk web 폴더 확인 (상위 디렉토리에서 찾기)
    kiosk_web_path = Path("../kiosk web")
    if kiosk_web_path.exists():
        print("✅ kiosk web UI")
        
        # build 폴더 확인
        build_path = kiosk_web_path / "build"
        if build_path.exists():
            print("✅ React build 폴더")
        else:
            print("⚠️  React build 폴더 (빌드 필요)")
    else:
        print("❌ kiosk web UI (폴더가 없습니다)")
        return False
    
    return True

def check_web_servers():
    """웹서버 연결 상태 확인"""
    print("\n🌐 웹서버 연결 상태 확인 중...")
    
    try:
        import requests
        import socket
        
        # 포트 사용 가능 여부 확인
        def is_port_available(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('localhost', port))
                    return True
                except OSError:
                    return False
        
        # 일반적인 웹서버 포트 확인
        common_ports = [3000, 3001, 8080, 8081, 8000, 8001]
        available_ports = [port for port in common_ports if is_port_available(port)]
        
        if available_ports:
            print(f"✅ 사용 가능한 포트: {available_ports[:3]}...")
        else:
            print("⚠️  사용 가능한 포트가 제한적입니다")
        
        print("✅ 웹서버 연결 준비 완료")
        
    except ImportError:
        print("⚠️  requests 모듈이 없어 상세 확인을 건너뜁니다")
    except Exception as e:
        print(f"⚠️  웹서버 연결 확인 중 오류: {e}")
    
    return True

def build_react_app():
    """React 앱 빌드"""
    print("\n🔨 React 앱 빌드 중...")
    
    kiosk_web_path = Path("../kiosk web")
    original_dir = os.getcwd()
    os.chdir(kiosk_web_path)
    
    # npm 경로 직접 지정 (환경 변수 문제 우회)
    npm_paths = [
        r"C:\Program Files\nodejs\npm.cmd",
        r"C:\Program Files (x86)\nodejs\npm.cmd",
        "npm"  # 마지막으로 기본 npm 시도
    ]
    
    npm_path = None
    for path in npm_paths:
        try:
            result = subprocess.run([path, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                npm_path = path
                print(f"✅ npm 경로 발견: {path}")
                break
        except FileNotFoundError:
            continue
    
    if not npm_path:
        print("❌ npm을 찾을 수 없습니다. Node.js가 설치되어 있는지 확인해주세요.")
        return False
    
    try:
        # npm install
        print("📦 npm install 실행 중...")
        subprocess.run([npm_path, "install"], check=True, capture_output=True)
        print("✅ npm install 완료")
        
        # npm run build
        print("🏗️  npm run build 실행 중...")
        subprocess.run([npm_path, "run", "build"], check=True, capture_output=True)
        print("✅ React 앱 빌드 완료")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ React 앱 빌드 실패: {e}")
        print(f"오류 출력: {e.stderr.decode() if e.stderr else '없음'}")
        return False
    except FileNotFoundError:
        print("❌ npm이 설치되지 않았습니다. Node.js를 설치해주세요.")
        return False
    finally:
        os.chdir(original_dir)
    
    return True

def run_scenario():
    """시나리오 실행"""
    print("\n🚀 헬 난이도 시나리오 실행 중...")
    
    try:
        from kiosk_scenario_collector import KioskScenarioCollector
        
        # 수집기 생성 및 실행
        collector = KioskScenarioCollector("R1_kiosk_hell")
        collector.run_scenario_collection()
        
    except ImportError as e:
        print(f"❌ 모듈 import 실패: {e}")
        return False
    except Exception as e:
        print(f"❌ 시나리오 실행 실패: {e}")
        return False
    
    return True

def main():
    """메인 함수"""
    print("🔥 헬 난이도 키오스크 시나리오 실행기")
    print("고령자 UI 사용성 연구를 위한 복잡한 시나리오 데이터 수집")
    print("="*70)
    
    # 의존성 확인
    if not check_dependencies():
        print("\n❌ 의존성 확인 실패. 필요한 패키지를 설치해주세요.")
        return
    
    # 웹서버 연결 확인
    if not check_web_servers():
        print("\n❌ 웹서버 연결 확인 실패.")
        return
    
    # React 앱 빌드
    if not build_react_app():
        print("\n❌ React 앱 빌드 실패.")
        return
    
    # 시나리오 실행
    print("\n🎯 모든 준비가 완료되었습니다!")
    print("시나리오를 시작합니다...")
    time.sleep(2)
    
    if not run_scenario():
        print("\n❌ 시나리오 실행 실패.")
        return
    
    print("\n✅ 시나리오가 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main() 