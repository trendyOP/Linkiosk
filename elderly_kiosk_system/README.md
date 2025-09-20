# 노인 친화적 키오스크 시스템

시선 추적 기반 적응형 UI와 머신러닝을 활용한 노인 친화적 키오스크 시스템입니다.

## 🎯 프로젝트 목적

노인들의 시선 추적 데이터를 분석하여 자동으로 UI를 조정하는 시스템을 구축합니다:
- **실시간 시선 추적**: 웹캠을 통한 시선 위치 감지
- **머신러닝 분석**: 사용자 유형 자동 분류 (젊은/중년/노년)
- **적응형 UI**: 사용자 유형에 따른 자동 UI 조정
- **노인 친화적 설계**: 큰 폰트, 높은 대비, 큰 버튼

## 📁 프로젝트 구조

```
elderly_kiosk_system/
├── elderly_ui_analyzer.py      # 시선 데이터 분석 및 머신러닝
├── adaptive_ui_system.py       # 실시간 적응형 UI 시스템
├── elderly_kiosk_system.py     # 메인 통합 시스템
├── requirements.txt            # 필요한 패키지 목록
└── README.md                  # 프로젝트 설명서
```

## 🚀 주요 기능

### 1. 시선 데이터 분석 (`elderly_ui_analyzer.py`)
- **데이터 로드**: 기존 노인 사용자 시선 추적 데이터 분석
- **패턴 분석**: 시선 이동 거리, 속도, 고정점 분석
- **머신러닝**: 랜덤 포레스트를 이용한 사용자 유형 분류
- **히트맵 생성**: 시선 집중 영역 시각화

### 2. 적응형 UI 시스템 (`adaptive_ui_system.py`)
- **실시간 시선 추적**: 웹캠 기반 시선 위치 감지
- **동적 UI 조정**: 사용자 유형에 따른 자동 UI 변경
- **고정점 분석**: 시선이 머무는 영역 실시간 분석
- **데이터 저장**: 사용자 상호작용 데이터 자동 저장

### 3. 통합 시스템 (`elderly_kiosk_system.py`)
- **실시간 분석**: 5초마다 사용자 유형 재분석
- **자동 UI 업데이트**: 신뢰도가 높을 때 UI 자동 변경
- **종합 리포트**: 사용 세션 종료 시 분석 리포트 생성

## 🎨 UI 적응 레벨

### 젊은 그룹 (Young)
- 폰트 크기: 16px
- 버튼 크기: 중간
- 색상: 표준
- 애니메이션: 빠름

### 중년 그룹 (Middle)
- 폰트 크기: 18px
- 버튼 크기: 큼
- 색상: 높은 대비
- 애니메이션: 중간

### 노년 그룹 (Elderly)
- 폰트 크기: 24px
- 버튼 크기: 매우 큼
- 색상: 노인 친화적
- 애니메이션: 느림

## 📊 분석 지표

### 시선 분석 지표
- **시선 이동 거리**: 픽셀 단위 평균 이동 거리
- **시선 속도**: 픽셀/초 단위 평균 이동 속도
- **고정점 수**: 시선이 머무는 영역 개수
- **고정 시간**: 각 영역에서 머무는 평균 시간

### 머신러닝 특성
- 평균 시선 이동 거리
- 평균 시선 속도
- 최대 시선 속도
- 고정점 수
- 평균 고정 시간
- 나이 정보

## 🛠️ 설치 및 실행

### 1. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터 분석 실행
```bash
python elderly_ui_analyzer.py
```

### 3. 적응형 UI 시스템 실행
```bash
python adaptive_ui_system.py
```

### 4. 통합 시스템 실행
```bash
python elderly_kiosk_system.py
```

## 📈 사용 예시

### 데이터 분석
```python
from elderly_ui_analyzer import ElderlyUIAnalyzer

# 분석기 초기화
analyzer = ElderlyUIAnalyzer()

# 데이터 로드 및 분석
analyzer.load_all_user_data()
analyzer.analyze_gaze_patterns()
analyzer.train_ml_model()

# 리포트 생성
report = analyzer.generate_report()
print(report)
```

### 실시간 UI 시스템
```python
from adaptive_ui_system import AdaptiveUIWidget
from PyQt5.QtWidgets import QApplication
import sys

# 애플리케이션 실행
app = QApplication(sys.argv)
ui = AdaptiveUIWidget(user_type="elderly")
ui.show()
sys.exit(app.exec_())
```

## 🔧 설정 옵션

### 시선 추적 설정
- **시뮬레이션 모드**: 실제 웹캠 없이 테스트 가능
- **실제 웹캠 모드**: Beam Eye Tracker SDK 연동
- **샘플링 주기**: 30 FPS (0.033초 간격)

### UI 적응 설정
- **자동 변경**: 신뢰도 70% 이상일 때 자동 변경
- **수동 변경**: UI 설정 변경 버튼으로 수동 변경
- **실시간 분석**: 5초마다 사용자 유형 재분석

## 📊 출력 파일

### 분석 결과
- `heatmap.png`: 시선 히트맵 이미지
- `gaze_data_*.json`: 실시간 시선 데이터
- `elderly_kiosk_report.txt`: 종합 분석 리포트

### 리포트 내용
- 기존 사용자 데이터 분석 결과
- 현재 시스템 상태
- UI 권장사항
- 시스템 개선 제안

## 🤝 기여 방법

1. 이슈 등록: 버그 리포트 또는 기능 제안
2. 브랜치 생성: 새로운 기능 개발
3. 테스트 실행: 기존 기능 동작 확인
4. 풀 리퀘스트: 코드 리뷰 후 병합

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 👥 개발팀

- **시선 추적 분석**: 머신러닝 기반 사용자 유형 분류
- **적응형 UI**: 실시간 UI 조정 시스템
- **노인 친화적 설계**: 접근성과 사용성 중심 설계

## 🔮 향후 계획

- [ ] 실제 웹캠 연동 구현
- [ ] 더 정교한 머신러닝 모델 개발
- [ ] 음성 명령 기능 추가
- [ ] 모바일 앱 버전 개발
- [ ] 다국어 지원
- [ ] 클라우드 기반 데이터 분석

---

**노인 친화적 키오스크 시스템**으로 모든 연령대가 편리하게 사용할 수 있는 키오스크를 만들어보세요! 🎯 