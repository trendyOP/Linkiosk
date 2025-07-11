#!/usr/bin/env python3
"""
수집된 데이터로 히트맵 생성 스크립트
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from datetime import datetime

def generate_heatmap_from_data(data_dir):
    """수집된 데이터로 히트맵 생성"""
    print(f"📁 데이터 디렉토리: {data_dir}")
    
    # 파일 경로
    gaze_file = os.path.join(data_dir, "gaze_data.json")
    screenshot_file = os.path.join(data_dir, "screenshot_metadata.json")
    
    # 파일 존재 확인
    if not os.path.exists(gaze_file):
        print(f"❌ 시선 데이터 파일이 없습니다: {gaze_file}")
        return
    
    if not os.path.exists(screenshot_file):
        print(f"❌ 스크린샷 메타데이터 파일이 없습니다: {screenshot_file}")
        return
    
    # 데이터 로드
    print("📊 데이터 로드 중...")
    with open(gaze_file, 'r', encoding='utf-8') as f:
        gaze_data = json.load(f)
    
    with open(screenshot_file, 'r', encoding='utf-8') as f:
        screenshot_data = json.load(f)
    
    # 시선 데이터 분석
    gaze_points = gaze_data['gaze_data']
    step_gaze_data = gaze_data.get('step_gaze_data', {})
    
    print(f"📈 시선 데이터: {len(gaze_points)}개")
    print(f"📊 단계별 데이터: {len(step_gaze_data)}개 단계")
    
    # matplotlib 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 전체 히트맵 생성
    print("🎨 전체 히트맵 생성 중...")
    create_overall_heatmap(gaze_points, data_dir)
    
    # 단계별 히트맵 생성
    print("🎨 단계별 히트맵 생성 중...")
    for step_num in step_gaze_data:
        step_gaze = step_gaze_data[step_num]
        if step_gaze:
            create_step_heatmap(step_num, step_gaze, data_dir)
    
    print("✅ 히트맵 생성 완료!")

def create_overall_heatmap(gaze_points, data_dir):
    """전체 히트맵 생성"""
    try:
        # 유효한 시선 데이터만 필터링 (confidence > 0)
        valid_gaze = [p for p in gaze_points if p.get('confidence', 0) > 0]
        
        if not valid_gaze:
            print("⚠️ 유효한 시선 데이터가 없습니다.")
            return
        
        # 좌표 추출
        x_coords = [p['x'] for p in valid_gaze]
        y_coords = [p['y'] for p in valid_gaze]
        
        # 히트맵 생성
        plt.figure(figsize=(16, 10))
        
        # 히트맵 생성
        h = plt.hist2d(x_coords, y_coords, bins=50, cmap='hot', alpha=0.7)
        
        # 컬러바 추가
        plt.colorbar(h[3], label='시선 빈도')
        
        # 제목과 레이블
        plt.title('전체 시선 히트맵 (모든 단계)', fontsize=14)
        plt.xlabel('X 좌표 (픽셀)')
        plt.ylabel('Y 좌표 (픽셀)')
        plt.grid(True, alpha=0.3)
        
        # 저장
        heatmap_file = os.path.join(data_dir, "overall_heatmap.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 전체 히트맵 저장: {heatmap_file}")
        
    except Exception as e:
        print(f"❌ 전체 히트맵 생성 실패: {e}")

def create_step_heatmap(step_num, step_gaze, data_dir):
    """단계별 히트맵 생성"""
    try:
        # 유효한 시선 데이터만 필터링
        valid_gaze = [p for p in step_gaze if p.get('confidence', 0) > 0]
        
        if not valid_gaze:
            print(f"⚠️ 단계 {step_num}의 유효한 시선 데이터가 없습니다.")
            return
        
        # 좌표 추출
        x_coords = [p['x'] for p in valid_gaze]
        y_coords = [p['y'] for p in valid_gaze]
        
        # 히트맵 생성
        plt.figure(figsize=(16, 10))
        
        # 히트맵 생성
        h = plt.hist2d(x_coords, y_coords, bins=30, cmap='hot', alpha=0.7)
        
        # 컬러바 추가
        plt.colorbar(h[3], label='시선 빈도')
        
        # 제목과 레이블
        plt.title(f'단계 {step_num + 1} 시선 히트맵', fontsize=14)
        plt.xlabel('X 좌표 (픽셀)')
        plt.ylabel('Y 좌표 (픽셀)')
        plt.grid(True, alpha=0.3)
        
        # 저장
        heatmap_file = os.path.join(data_dir, f"step_{step_num + 1}_heatmap.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 단계 {step_num + 1} 히트맵 저장: {heatmap_file}")
        
    except Exception as e:
        print(f"❌ 단계 {step_num} 히트맵 생성 실패: {e}")

def main():
    """메인 함수"""
    print("🔥 수집된 데이터로 히트맵 생성")
    print("="*50)
    
    # 데이터 디렉토리 지정
    data_dir = "kiosk_data/R1_kiosk_hell_enhanced_20250630_000439"
    
    if not os.path.exists(data_dir):
        print(f"❌ 데이터 디렉토리가 없습니다: {data_dir}")
        return
    
    # 히트맵 생성
    generate_heatmap_from_data(data_dir)

if __name__ == "__main__":
    main() 