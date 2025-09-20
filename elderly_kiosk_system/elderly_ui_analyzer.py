import json
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import cv2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Windows에서 joblib 병렬 처리 문제 해결
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

class ElderlyUIAnalyzer:
    def __init__(self, data_path: str = "../beameyetracker/kiosk_data"):
        self.data_path = data_path
        self.user_data = {}
        self.analysis_results = {}
        self.ml_model = None
        self.scaler = StandardScaler()
        
    def load_all_user_data(self) -> Dict:
        """모든 노인 사용자 데이터를 로드합니다."""
        print("노인 사용자 데이터 로딩 중...")
        
        loaded_count = 0
        for folder in os.listdir(self.data_path):
            if folder.startswith("uiux_certification_") and "year" in folder:
                # 폴더명에서 년생 정보 추출
                parts = folder.split("_")
                user_id = None
                
                # 폴더명 전체를 고유 ID로 사용
                unique_id = folder
                
                # 년생 정보 추출 (표시용)
                user_id = None
                for part in parts:
                    if "year" in part:
                        user_id = part
                        break
                
                if not user_id:
                    continue
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
                    
                    # 년생에서 나이 계산 (2025년 기준)
                    try:
                        # 다양한 형식 처리
                        if '-' in user_id:
                            # "163430-44year" 같은 형식
                            birth_year = int(user_id.split('-')[-1].replace('year', ''))
                        else:
                            # "43year" 같은 형식
                            birth_year = int(user_id.replace('year', ''))
                        
                        # 년생이 1900년대인 경우 (예: 43년생 -> 1943년)
                        if birth_year < 100:
                            birth_year = 1900 + birth_year
                        
                        age = 2025 - birth_year
                    except ValueError:
                        print(f"년생 파싱 실패: {user_id}")
                        continue
                    
                    # 고유 ID 사용
                    self.user_data[unique_id] = {
                        'gaze_data': gaze_data,
                        'click_data': click_data,
                        'performance': performance_data,
                        'scenario': scenario_data,
                        'age': age,
                        'birth_year': birth_year,
                        'original_folder': folder
                    }
                    
                    print(f"사용자 {user_id} ({folder}) 데이터 로드 완료")
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"사용자 {user_id} 데이터 로드 실패: {e}")
        
        print(f"총 {loaded_count}명의 사용자 데이터 로드 완료")
        return self.user_data
    
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
                    
                    # 시간 간격 계산 (타임스탬프 차이)
                    dt = gaze_data[i]['timestamp'] - gaze_data[i-1]['timestamp']
                    if dt > 0:
                        velocity = distance / dt
                        velocities.append(velocity)
            
            # 고정점(fixation) 분석
            fixations = self._analyze_fixations(gaze_data)
            
            # 시선 클러스터링
            gaze_points = np.array([[x, y] for x, y in zip(x_coords, y_coords) if x > 0 and y > 0])
            if len(gaze_points) > 0:
                clusters = self._cluster_gaze_points(gaze_points)
            else:
                clusters = []
            
            self.analysis_results[user_id] = {
                'age': data['age'],
                'birth_year': data.get('birth_year', 'N/A'),
                'gaze_statistics': {
                    'total_points': len(gaze_data),
                    'valid_points': len([p for p in gaze_data if p['confidence'] > 0]),
                    'average_distance': np.mean(distances) if distances else 0,
                    'average_velocity': np.mean(velocities) if velocities else 0,
                    'max_velocity': np.max(velocities) if velocities else 0,
                    'fixation_count': len(fixations),
                    'average_fixation_duration': np.mean([f['duration'] for f in fixations]) if fixations else 0
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
    
    def train_ml_model(self) -> RandomForestClassifier:
        """머신러닝 모델을 훈련합니다."""
        print("머신러닝 모델 훈련 중...")
        
        # 특성 추출
        features = []
        labels = []
        
        for user_id, analysis in self.analysis_results.items():
            stats = analysis['gaze_statistics']
            
            # 특성 벡터 생성
            feature_vector = [
                stats['average_distance'],
                stats['average_velocity'],
                stats['max_velocity'],
                stats['fixation_count'],
                stats['average_fixation_duration'],
                analysis['age']
            ]
            
            features.append(feature_vector)
            
            # 레이블 생성 (나이대별 그룹)
            age = analysis['age']
            if age < 45:
                label = 0  # 젊은 그룹
            elif age < 50:
                label = 1  # 중년 그룹
            else:
                label = 2  # 노년 그룹
            
            labels.append(label)
        
        # 데이터 정규화
        features_scaled = self.scaler.fit_transform(features)
        
        # 랜덤 포레스트 모델 훈련
        self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        self.ml_model.fit(features_scaled, labels)
        
        print("머신러닝 모델 훈련 완료")
        return self.ml_model
    
    def predict_user_type(self, gaze_features: List[float]) -> Dict:
        """사용자 유형을 예측합니다."""
        if self.ml_model is None:
            raise ValueError("모델이 훈련되지 않았습니다. train_ml_model()을 먼저 호출하세요.")
        
        features_scaled = self.scaler.transform([gaze_features])
        prediction = self.ml_model.predict(features_scaled)[0]
        
        user_types = {
            0: "young",      # 젊은 그룹
            1: "middle",     # 중년 그룹  
            2: "elderly"     # 노년 그룹
        }
        
        return {
            'user_type': user_types[prediction],
            'confidence': np.max(self.ml_model.predict_proba(features_scaled))
        }
    
    def generate_ui_recommendations(self, user_type: str) -> Dict:
        """사용자 유형에 따른 UI 권장사항을 생성합니다."""
        recommendations = {
            'young': {
                'font_size': 16,
                'button_size': 'medium',
                'color_scheme': 'standard',
                'animation_speed': 'fast',
                'text_contrast': 'normal'
            },
            'middle': {
                'font_size': 18,
                'button_size': 'large',
                'color_scheme': 'high_contrast',
                'animation_speed': 'medium',
                'text_contrast': 'high'
            },
            'elderly': {
                'font_size': 24,
                'button_size': 'extra_large',
                'color_scheme': 'elderly_friendly',
                'animation_speed': 'slow',
                'text_contrast': 'very_high'
            }
        }
        
        return recommendations.get(user_type, recommendations['elderly'])
    
    def create_heatmap(self, gaze_data: List[Dict], output_path: str = "heatmap.png"):
        """시선 히트맵을 생성합니다."""
        # 유효한 시선 데이터만 추출
        valid_points = [(point['x'], point['y']) for point in gaze_data if point['confidence'] > 0 and point['x'] > 0 and point['y'] > 0]
        
        if not valid_points:
            print("유효한 시선 데이터가 없습니다.")
            return
        
        x_coords, y_coords = zip(*valid_points)
        
        # 히트맵 생성
        plt.figure(figsize=(12, 8))
        plt.hist2d(x_coords, y_coords, bins=50, cmap='hot')
        plt.colorbar(label='시선 빈도')
        plt.title('사용자 시선 히트맵')
        plt.xlabel('X 좌표')
        plt.ylabel('Y 좌표')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"히트맵이 {output_path}에 저장되었습니다.")
    
    def generate_report(self) -> str:
        """분석 결과 리포트를 생성합니다."""
        report = []
        report.append("=" * 50)
        report.append("노인 사용자 시선 추적 분석 리포트")
        report.append("=" * 50)
        report.append("")
        
        # 전체 통계
        total_loaded = len(self.user_data)  # 실제 로드된 사용자 수
        total_analyzed = len(self.analysis_results)  # 분석 완료된 사용자 수
        ages = [data['age'] for data in self.user_data.values()]
        avg_age = np.mean(ages)
        
        report.append(f"총 로드된 사용자 수: {total_loaded}명")
        report.append(f"분석 완료된 사용자 수: {total_analyzed}명")
        report.append(f"평균 나이: {avg_age:.1f}세")
        report.append(f"나이 범위: {min(ages)}세 ~ {max(ages)}세")
        report.append("")
        
        # 사용자별 상세 분석
        report.append("📊 분석 완료된 사용자:")
        for unique_id, data in self.user_data.items():
            if unique_id in self.analysis_results:
                stats = self.analysis_results[unique_id]['gaze_statistics']
                birth_year = data.get('birth_year', 'N/A')
                original_folder = data.get('original_folder', unique_id)
                # 폴더명에서 년생 정보 추출
                year_info = None
                for part in original_folder.split('_'):
                    if 'year' in part:
                        year_info = part
                        break
                
                report.append(f"✅ 사용자 {year_info} ({data['age']}세, {birth_year}년생) - {original_folder}:")
                report.append(f"  - 총 시선 점 수: {stats['total_points']}")
                report.append(f"  - 유효한 시선 점 수: {stats['valid_points']}")
                report.append(f"  - 평균 시선 이동 거리: {stats['average_distance']:.2f} 픽셀")
                report.append(f"  - 평균 시선 속도: {stats['average_velocity']:.2f} 픽셀/초")
                report.append(f"  - 고정점 수: {stats['fixation_count']}")
                report.append(f"  - 평균 고정 시간: {stats['average_fixation_duration']:.2f}초")
                report.append("")
        
        # 분석 실패한 사용자들
        failed_users = [unique_id for unique_id in self.user_data.keys() if unique_id not in self.analysis_results]
        if failed_users:
            report.append("❌ 분석 실패한 사용자:")
            for unique_id in failed_users:
                data = self.user_data[unique_id]
                birth_year = data.get('birth_year', 'N/A')
                original_folder = data.get('original_folder', unique_id)
                # 폴더명에서 년생 정보 추출
                year_info = None
                for part in original_folder.split('_'):
                    if 'year' in part:
                        year_info = part
                        break
                
                report.append(f"❌ 사용자 {year_info} ({data['age']}세, {birth_year}년생) - {original_folder}: 분석 실패")
            report.append("")
        
        return "\n".join(report)

# 사용 예시
if __name__ == "__main__":
    analyzer = ElderlyUIAnalyzer()
    
    # 데이터 로드
    analyzer.load_all_user_data()
    
    # 시선 패턴 분석
    analyzer.analyze_gaze_patterns()
    
    # 머신러닝 모델 훈련
    analyzer.train_ml_model()
    
    # 리포트 생성
    report = analyzer.generate_report()
    print(report)
    
    # 예시: 새로운 사용자 유형 예측
    example_features = [200.0, 500.0, 8000.0, 15, 1.5, 45]  # 예시 특성
    prediction = analyzer.predict_user_type(example_features)
    print(f"예측된 사용자 유형: {prediction['user_type']}")
    print(f"신뢰도: {prediction['confidence']:.2f}")
    
    # UI 권장사항 생성
    recommendations = analyzer.generate_ui_recommendations(prediction['user_type'])
    print(f"UI 권장사항: {recommendations}") 