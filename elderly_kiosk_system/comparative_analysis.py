#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
사용자 데이터와 노인 데이터 비교 분석
클러스터링을 통한 사용자 그룹 분류
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # 백엔드를 Agg로 설정
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Windows에서 joblib 병렬 처리 문제 해결
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

class ComparativeAnalyzer:
    def __init__(self, data_path: str = "../beameyetracker/kiosk_data"):
        self.data_path = data_path
        self.user_data = {}
        self.analysis_results = {}
        self.scaler = StandardScaler()
        
    def load_all_data(self) -> Dict:
        """모든 사용자 데이터를 로드합니다."""
        print("사용자 데이터 로딩 중...")
        
        loaded_count = 0
        for folder in os.listdir(self.data_path):
            if folder.startswith("uiux_certification_"):
                # 폴더명에서 사용자 정보 추출
                parts = folder.split("_")
                user_id = None
                
                # 폴더명 전체를 고유 ID로 사용
                unique_id = folder
                
                # 사용자 유형 분류
                user_type = self.classify_user_type(folder)
                
                folder_path = os.path.join(self.data_path, folder)
                
                try:
                    # 각 데이터 파일 로드
                    with open(os.path.join(folder_path, "gaze_data.json"), 'r', encoding='utf-8') as f:
                        gaze_data = json.load(f)
                    
                    with open(os.path.join(folder_path, "click_events.json"), 'r', encoding='utf-8') as f:
                        click_data = json.load(f)
                    
                    with open(os.path.join(folder_path, "performance_metrics.json"), 'r', encoding='utf-8') as f:
                        performance_data = json.load(f)
                    
                    with open(os.path.join(folder_path, "scenario_data.json"), 'r', encoding='utf-8') as f:
                        scenario_data = json.load(f)
                    
                    # 나이 추정 (폴더명 기반)
                    estimated_age = self.estimate_age_from_folder(folder)
                    
                    self.user_data[unique_id] = {
                        'gaze_data': gaze_data,
                        'click_data': click_data,
                        'performance': performance_data,
                        'scenario': scenario_data,
                        'user_type': user_type,
                        'estimated_age': estimated_age,
                        'original_folder': folder
                    }
                    
                    print(f"사용자 {user_type} ({folder}) 데이터 로드 완료")
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"사용자 {folder} 데이터 로드 실패: {e}")
        
        print(f"총 {loaded_count}명의 사용자 데이터 로드 완료")
        return self.user_data
    
    def classify_user_type(self, folder_name: str) -> str:
        """폴더명을 기반으로 사용자 유형 분류"""
        if "20250731" in folder_name and "year" in folder_name:
            return "elderly"  # 노인 데이터
        elif "20250719" in folder_name:
            return "young"     # 당신의 데이터
        else:
            return "unknown"
    
    def estimate_age_from_folder(self, folder_name: str) -> int:
        """폴더명에서 나이 추정"""
        if "20250731" in folder_name and "year" in folder_name:
            # 노인 데이터: 년생에서 나이 계산
            for part in folder_name.split("_"):
                if "year" in part:
                    try:
                        birth_year = int(part.replace("year", ""))
                        if birth_year < 100:
                            birth_year = 1900 + birth_year
                        return 2025 - birth_year
                    except:
                        pass
            return 75  # 기본 노인 나이
        elif "20250719" in folder_name:
            return 25  # 당신의 추정 나이
        else:
            return 30  # 기본 나이
    
    def analyze_gaze_patterns(self) -> Dict:
        """시선 패턴을 분석합니다."""
        print("시선 패턴 분석 중...")
        
        for user_id, data in self.user_data.items():
            gaze_data = data['gaze_data']
            
            # 시선 좌표 추출
            x_coords = [point['x'] for point in gaze_data if point['confidence'] > 0]
            y_coords = [point['y'] for point in gaze_data if point['confidence'] > 0]
            
            # 시선 이동 거리 계산
            distances = []
            velocities = []
            
            for i in range(1, len(gaze_data)):
                if gaze_data[i]['confidence'] > 0 and gaze_data[i-1]['confidence'] > 0:
                    dx = gaze_data[i]['x'] - gaze_data[i-1]['x']
                    dy = gaze_data[i]['y'] - gaze_data[i-1]['y']
                    distance = np.sqrt(dx**2 + dy**2)
                    distances.append(distance)
                    
                    # 시간 간격 계산
                    dt = gaze_data[i]['timestamp'] - gaze_data[i-1]['timestamp']
                    if dt > 0:
                        velocity = distance / dt
                        velocities.append(velocity)
            
            # 고정점 분석
            fixations = self._analyze_fixations(gaze_data)
            
            # 시선 클러스터링
            gaze_points = np.array([[x, y] for x, y in zip(x_coords, y_coords) if x > 0 and y > 0])
            if len(gaze_points) > 0:
                clusters = self._cluster_gaze_points(gaze_points)
            else:
                clusters = []
            
            self.analysis_results[user_id] = {
                'user_type': data['user_type'],
                'estimated_age': data['estimated_age'],
                'gaze_statistics': {
                    'total_points': len(gaze_data),
                    'valid_points': len([p for p in gaze_data if p['confidence'] > 0]),
                    'average_distance': np.mean(distances) if distances else 0,
                    'average_velocity': np.mean(velocities) if velocities else 0,
                    'max_velocity': np.max(velocities) if velocities else 0,
                    'fixation_count': len(fixations),
                    'average_fixation_duration': np.mean([f['duration'] for f in fixations]) if fixations else 0,
                    'gaze_efficiency': len([p for p in gaze_data if p['confidence'] > 0]) / len(gaze_data) if gaze_data else 0
                },
                'gaze_clusters': clusters,
                'fixations': fixations
            }
        
        return self.analysis_results
    
    def _analyze_fixations(self, gaze_data: List[Dict], threshold: float = 30.0, min_duration: float = 0.1) -> List[Dict]:
        """고정점(fixation)을 분석합니다."""
        fixations = []
        current_fixation = None
        
        for i, point in enumerate(gaze_data):
            if point['confidence'] == 0:
                continue
                
            if current_fixation is None:
                current_fixation = {
                    'start_time': point['timestamp'],
                    'start_x': point['x'],
                    'start_y': point['y'],
                    'points': [point]
                }
            else:
                # 현재 점과 고정점 중심점 간의 거리 계산
                center_x = np.mean([p['x'] for p in current_fixation['points']])
                center_y = np.mean([p['y'] for p in current_fixation['points']])
                
                distance = np.sqrt((point['x'] - center_x)**2 + (point['y'] - center_y)**2)
                
                if distance <= threshold:
                    # 고정점에 추가
                    current_fixation['points'].append(point)
                else:
                    # 고정점 종료 및 저장
                    duration = point['timestamp'] - current_fixation['start_time']
                    if duration >= min_duration:
                        current_fixation['duration'] = duration
                        current_fixation['center_x'] = center_x
                        current_fixation['center_y'] = center_y
                        fixations.append(current_fixation)
                    
                    # 새로운 고정점 시작
                    current_fixation = {
                        'start_time': point['timestamp'],
                        'start_x': point['x'],
                        'start_y': point['y'],
                        'points': [point]
                    }
        
        return fixations
    
    def _cluster_gaze_points(self, gaze_points: np.ndarray, n_clusters: int = 5) -> List[Dict]:
        """시선 점들을 클러스터링합니다."""
        if len(gaze_points) < n_clusters:
            n_clusters = len(gaze_points)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(gaze_points)
        
        cluster_info = []
        for i in range(n_clusters):
            cluster_points = gaze_points[clusters == i]
            cluster_info.append({
                'cluster_id': i,
                'center_x': float(kmeans.cluster_centers_[i][0]),
                'center_y': float(kmeans.cluster_centers_[i][1]),
                'point_count': len(cluster_points),
                'density': len(cluster_points) / len(gaze_points)
            })
        
        return cluster_info
    
    def perform_clustering_analysis(self, n_clusters: int = 3):
        """클러스터링 분석을 수행합니다."""
        print("클러스터링 분석 수행 중...")
        
        # 특성 추출
        features = []
        user_ids = []
        user_types = []
        
        for user_id, analysis in self.analysis_results.items():
            stats = analysis['gaze_statistics']
            
            # 특성 벡터 생성
            feature_vector = [
                stats['average_distance'],
                stats['average_velocity'],
                stats['max_velocity'],
                stats['fixation_count'],
                stats['average_fixation_duration'],
                stats['gaze_efficiency'],
                analysis['estimated_age']
            ]
            
            features.append(feature_vector)
            user_ids.append(user_id)
            user_types.append(analysis['user_type'])
        
        if len(features) < n_clusters:
            print("데이터가 부족하여 클러스터링을 수행할 수 없습니다.")
            return None
        
        # 데이터 정규화
        features_scaled = self.scaler.fit_transform(features)
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # PCA를 통한 2D 시각화
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)
        
        # 주성분 분석 결과 저장
        self.pca_components = pca.components_
        self.pca_explained_variance = pca.explained_variance_ratio_
        
        # 결과 저장
        clustering_results = {
            'user_ids': user_ids,
            'user_types': user_types,
            'cluster_labels': cluster_labels,
            'features_2d': features_2d,
            'features_scaled': features_scaled,
            'cluster_centers': kmeans.cluster_centers_
        }
        
        return clustering_results
    
    def generate_comparative_report(self, clustering_results: Dict) -> str:
        """비교 분석 리포트를 생성합니다."""
        report = []
        report.append("=" * 60)
        report.append("사용자 데이터 vs 노인 데이터 비교 분석 리포트")
        report.append("=" * 60)
        report.append("")
        
        # 전체 통계
        total_users = len(self.user_data)
        user_type_counts = {}
        for data in self.user_data.values():
            user_type = data['user_type']
            user_type_counts[user_type] = user_type_counts.get(user_type, 0) + 1
        
        report.append("📊 전체 데이터 현황:")
        report.append(f"총 사용자 수: {total_users}명")
        for user_type, count in user_type_counts.items():
            report.append(f"  - {user_type}: {count}명")
        report.append("")
        
        # 사용자 유형별 평균 통계
        report.append("📈 사용자 유형별 평균 통계:")
        user_type_stats = {}
        
        for user_type in set(user_type_counts.keys()):
            type_users = [analysis for analysis in self.analysis_results.values() 
                         if analysis['user_type'] == user_type]
            
            if type_users:
                avg_distance = np.mean([u['gaze_statistics']['average_distance'] for u in type_users])
                avg_velocity = np.mean([u['gaze_statistics']['average_velocity'] for u in type_users])
                avg_fixations = np.mean([u['gaze_statistics']['fixation_count'] for u in type_users])
                avg_efficiency = np.mean([u['gaze_statistics']['gaze_efficiency'] for u in type_users])
                
                user_type_stats[user_type] = {
                    'avg_distance': avg_distance,
                    'avg_velocity': avg_velocity,
                    'avg_fixations': avg_fixations,
                    'avg_efficiency': avg_efficiency
                }
                
                report.append(f"🔹 {user_type.upper()} 그룹:")
                report.append(f"  - 평균 시선 이동 거리: {avg_distance:.2f} 픽셀")
                report.append(f"  - 평균 시선 속도: {avg_velocity:.2f} 픽셀/초")
                report.append(f"  - 평균 고정점 수: {avg_fixations:.1f}")
                report.append(f"  - 평균 시선 효율성: {avg_efficiency:.3f}")
                report.append("")
        
        # 주성분 분석 결과
        if hasattr(self, 'pca_explained_variance'):
            report.append("📊 주성분 분석 결과:")
            report.append(f"  - 주성분 1 설명 분산: {self.pca_explained_variance[0]:.1%}")
            report.append(f"  - 주성분 2 설명 분산: {self.pca_explained_variance[1]:.1%}")
            report.append(f"  - 총 설명 분산: {sum(self.pca_explained_variance):.1%}")
            report.append("")
            report.append("🔍 주성분 의미:")
            report.append("  - 주성분 1: 시선 이동 거리와 속도를 반영한 '시선 활동성'")
            report.append("  - 주성분 2: 고정점 수와 효율성을 반영한 '시선 안정성'")
            report.append("")
        
        # 클러스터링 결과
        if clustering_results:
            report.append("🎯 클러스터링 분석 결과:")
            
            cluster_info = {}
            for i, label in enumerate(clustering_results['cluster_labels']):
                user_type = clustering_results['user_types'][i]
                if label not in cluster_info:
                    cluster_info[label] = {'young': 0, 'elderly': 0, 'unknown': 0}
                cluster_info[label][user_type] += 1
            
            for cluster_id, counts in cluster_info.items():
                total_in_cluster = sum(counts.values())
                young_ratio = counts['young'] / total_in_cluster if total_in_cluster > 0 else 0
                elderly_ratio = counts['elderly'] / total_in_cluster if total_in_cluster > 0 else 0
                
                report.append(f"🔸 클러스터 {cluster_id + 1}:")
                report.append(f"  - 총 사용자: {total_in_cluster}명")
                report.append(f"  - 젊은 그룹 비율: {young_ratio:.1%}")
                report.append(f"  - 노인 그룹 비율: {elderly_ratio:.1%}")
                
                if young_ratio > 0.7:
                    report.append(f"  - 특성: 젊은 사용자 중심 그룹")
                elif elderly_ratio > 0.7:
                    report.append(f"  - 특성: 노인 사용자 중심 그룹")
                else:
                    report.append(f"  - 특성: 혼합 그룹")
                report.append("")
        
        # 주요 발견사항
        report.append("🔍 주요 발견사항:")
        
        if 'young' in user_type_stats and 'elderly' in user_type_stats:
            young_stats = user_type_stats['young']
            elderly_stats = user_type_stats['elderly']
            
            # 시선 이동 거리 비교
            if elderly_stats['avg_distance'] > young_stats['avg_distance']:
                report.append("✅ 노인 그룹이 젊은 그룹보다 시선 이동 거리가 큽니다.")
            else:
                report.append("✅ 젊은 그룹이 노인 그룹보다 시선 이동 거리가 큽니다.")
            
            # 시선 속도 비교
            if elderly_stats['avg_velocity'] > young_stats['avg_velocity']:
                report.append("✅ 노인 그룹이 젊은 그룹보다 시선 속도가 빠릅니다.")
            else:
                report.append("✅ 젊은 그룹이 노인 그룹보다 시선 속도가 빠릅니다.")
            
            # 고정점 수 비교
            if elderly_stats['avg_fixations'] > young_stats['avg_fixations']:
                report.append("✅ 노인 그룹이 젊은 그룹보다 고정점이 많습니다.")
            else:
                report.append("✅ 젊은 그룹이 노인 그룹보다 고정점이 많습니다.")
            
            # 시선 효율성 비교
            if elderly_stats['avg_efficiency'] > young_stats['avg_efficiency']:
                report.append("✅ 노인 그룹이 젊은 그룹보다 시선 효율성이 높습니다.")
            else:
                report.append("✅ 젊은 그룹이 노인 그룹보다 시선 효율성이 높습니다.")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def create_visualization(self, clustering_results: Dict):
        """시각화를 생성합니다."""
        if not clustering_results:
            return
        
        # 한글 폰트 설정
        try:
            # Windows에서 기본 한글 폰트 찾기
            font_path = None
            for font in fm.findSystemFonts():
                if 'malgun' in font.lower() or 'gulim' in font.lower():
                    font_path = font
                    break
            
            if font_path:
                plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
                print(f"한글 폰트 설정: {font_path}")
            else:
                # 기본 한글 폰트 시도
                plt.rcParams['font.family'] = 'Malgun Gothic'
        except:
            plt.rcParams['font.family'] = 'Malgun Gothic'
        
        # 2D 산점도 생성
        plt.figure(figsize=(15, 10))
        
        # 서브플롯 1: 클러스터별 분포
        plt.subplot(2, 2, 1)
        features_2d = clustering_results['features_2d']
        cluster_labels = clustering_results['cluster_labels']
        user_types = clustering_results['user_types']
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, cluster_id in enumerate(set(cluster_labels)):
            mask = cluster_labels == cluster_id
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=colors[i], marker=markers[i], s=100, alpha=0.7,
                       label=f'클러스터 {cluster_id + 1}')
        
        plt.title('K-means 클러스터링 결과 (3개 그룹)')
        plt.xlabel('주성분 1 (시선 이동 거리 + 속도)')
        plt.ylabel('주성분 2 (고정점 수 + 효율성)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 2: 사용자 유형별 분포
        plt.subplot(2, 2, 2)
        young_mask = [t == 'young' for t in user_types]
        elderly_mask = [t == 'elderly' for t in user_types]
        
        plt.scatter(features_2d[young_mask, 0], features_2d[young_mask, 1], 
                   c='blue', marker='o', s=100, alpha=0.7, label='젊은 그룹')
        plt.scatter(features_2d[elderly_mask, 0], features_2d[elderly_mask, 1], 
                   c='red', marker='s', s=100, alpha=0.7, label='노인 그룹')
        
        plt.title('사용자 그룹 비교')
        plt.xlabel('주성분 1 (시선 이동 거리 + 속도)')
        plt.ylabel('주성분 2 (고정점 수 + 효율성)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 3: 특성 비교 히스토그램
        plt.subplot(2, 2, 3)
        young_features = []
        elderly_features = []
        
        for user_id, analysis in self.analysis_results.items():
            if analysis['user_type'] == 'young':
                young_features.append(analysis['estimated_age'])
            elif analysis['user_type'] == 'elderly':
                elderly_features.append(analysis['estimated_age'])
        
        plt.hist(young_features, alpha=0.7, label='젊은 그룹', bins=10, color='blue')
        plt.hist(elderly_features, alpha=0.7, label='노인 그룹', bins=10, color='red')
        plt.title('나이 분포 비교')
        plt.xlabel('추정 나이 (세)')
        plt.ylabel('사용자 수')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 서브플롯 4: 시선 효율성 비교
        plt.subplot(2, 2, 4)
        young_efficiency = []
        elderly_efficiency = []
        
        for user_id, analysis in self.analysis_results.items():
            if analysis['user_type'] == 'young':
                young_efficiency.append(analysis['gaze_statistics']['gaze_efficiency'])
            elif analysis['user_type'] == 'elderly':
                elderly_efficiency.append(analysis['gaze_statistics']['gaze_efficiency'])
        
        plt.boxplot([young_efficiency, elderly_efficiency], 
                   labels=['젊은 그룹', '노인 그룹'])
        plt.title('시선 효율성 비교')
        plt.ylabel('시선 효율성')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comparative_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # PIL을 사용해서 한글 텍스트 추가
        self.add_korean_text_to_image('comparative_analysis.png')
        
        print("시각화가 'comparative_analysis.png'에 저장되었습니다.")
    
    def add_korean_text_to_image(self, image_path: str):
        """PIL을 사용해서 이미지에 한글 텍스트를 추가합니다."""
        try:
            # 이미지 로드
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # 한글 폰트 찾기
            font_size = 20
            try:
                # Windows 기본 한글 폰트
                font = ImageFont.truetype("malgun.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("gulim.ttc", font_size)
                except:
                    # 기본 폰트 사용
                    font = ImageFont.load_default()
            
            # 한글 텍스트 추가
            texts = [
                ("K-means 클러스터링 결과 (3개 그룹)", (50, 30)),
                ("사용자 그룹 비교 (젊은 그룹 vs 노인 그룹)", (50, 280)),
                ("나이 분포 비교", (50, 530)),
                ("시선 효율성 비교", (50, 780))
            ]
            
            for text, position in texts:
                draw.text(position, text, fill=(0, 0, 0), font=font)
            
            # 수정된 이미지 저장
            img.save(image_path)
            
        except Exception as e:
            print(f"한글 텍스트 추가 중 오류: {e}")

# 사용 예시
if __name__ == "__main__":
    analyzer = ComparativeAnalyzer()
    
    # 데이터 로드
    analyzer.load_all_data()
    
    # 시선 패턴 분석
    analyzer.analyze_gaze_patterns()
    
    # 클러스터링 분석
    clustering_results = analyzer.perform_clustering_analysis(n_clusters=3)
    
    # 리포트 생성
    report = analyzer.generate_comparative_report(clustering_results)
    print(report)
    
    # 시각화 생성
    analyzer.create_visualization(clustering_results)
    
    # 리포트를 파일로 저장
    with open("comparative_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n📄 비교 분석 리포트가 'comparative_analysis_report.txt'에 저장되었습니다.") 