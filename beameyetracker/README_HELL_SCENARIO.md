# 🔥 헬 난이도 키오스크 시나리오 데이터 수집기

고령자 UI 사용성 연구를 위한 복잡한 시나리오 데이터 수집 시스템입니다.

## 📋 개요

이 시스템은 고령자가 복잡한 키오스크 UI를 사용할 때의 시선 패턴과 실패 지점을 분석하기 위해 설계되었습니다.

### 🎯 주요 특징

- **복잡한 카테고리 구조**: 10개의 다양한 메뉴 카테고리
- **다양한 난이도**: easy, medium, hard, very_hard 단계별 난이도
- **실시간 진행 표시**: 웹 기반 실시간 시나리오 진행 상황 표시
- **자동 실패 추적**: 사용자가 어디서 실패했는지 자동 기록
- **시선 추적 통합**: Eyeware SDK를 통한 정확한 시선 데이터 수집

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
# Python 패키지 설치
pip install numpy matplotlib requests keyboard

# Node.js 설치 (React 앱 빌드용)
# https://nodejs.org/ 에서 다운로드
```

### 2. 실행

```bash
# 자동 실행 스크립트 사용 (권장)
python run_hell_scenario.py

# 또는 직접 실행
python kiosk_scenario_collector.py
```

## 📊 시나리오 구조

### 단계별 난이도

| 단계 | 카테고리 | 대상 메뉴 | 난이도 | 설명 |
|------|----------|-----------|--------|------|
| 1 | 베스트·신메뉴 | 흑당 콜드브루 | easy | 첫 번째 카테고리 |
| 2 | 커피 | 카라멜 마끼아또 | medium | 두 번째 카테고리 |
| 3 | 논-커피 음료 | 민트 초코 라떼 | hard | 세 번째 카테고리 |
| 4 | 티·에이드 | 샤인머스캣 에이드 | hard | 네 번째 카테고리 |
| 5 | 디저트 | 뉴욕 치즈케이크 | medium | 다섯 번째 카테고리 |
| 6 | 베이커리·샌드위치 | 버터 크루아상 | hard | 여섯 번째 카테고리 |
| 7 | 프라푸치노·블렌디드 | 자바칩 프라푸치노 | hard | 일곱 번째 카테고리 |
| 8 | 스무디·주스 | 블루베리 요거트 스무디 | hard | 여덟 번째 카테고리 |
| 9 | 빙수 | 딸기 빙수 | very_hard | 아홉 번째 카테고리 |
| 10 | 시즌 스페셜 | 펌킨 스파이스 라떼 | very_hard | 열 번째 카테고리 |
| 11 | 장바구니 | 장바구니 보기 | easy | 결제 단계 |
| 12 | 결제 | 결제하기 | medium | 결제 진행 |
| 13 | 결제수단 | 신용카드 | medium | 결제 방법 선택 |
| 14 | 식사방식 | 매장 | easy | 최종 선택 |

## ⌨️ 키보드 단축키

| 단축키 | 기능 | 설명 |
|--------|------|------|
| `Ctrl+Shift+S` | 다음 단계 진행 | 현재 단계를 완료하고 다음 단계로 진행 |
| `Ctrl+Shift+F` | 실패 기록 | 현재 단계를 실패로 기록 |
| `Ctrl+Shift+R` | 시나리오 재시작 | 전체 시나리오를 처음부터 다시 시작 |

## 📁 데이터 구조

수집된 데이터는 `kiosk_data/R1_kiosk_hell_YYYYMMDD_HHMMSS/` 폴더에 저장됩니다.

### 파일 목록

- `collection.log`: 전체 수집 과정 로그
- `gaze_data.json`: 시선 추적 데이터 (JSON)
- `gaze_data.csv`: 시선 추적 데이터 (CSV)
- `click_events.json`: 클릭 이벤트 데이터 (JSON)
- `click_events.csv`: 클릭 이벤트 데이터 (CSV)
- `error_logs.json`: 오류 및 실패 기록 (JSON)
- `error_logs.csv`: 오류 및 실패 기록 (CSV)
- `gaze_heatmap.png`: 시선 히트맵 시각화
- `gaze_trajectory.png`: 시선 궤적 시각화
- `failure_gaze_step_X.json`: 단계별 실패 시 시선 데이터

## 🔧 설정 및 커스터마이징

### 시나리오 수정

`kiosk_scenario_collector.py`의 `scenarios` 딕셔너리를 수정하여 새로운 시나리오를 정의할 수 있습니다.

```python
self.scenarios = {
    "custom_scenario": [
        {
            "step": 1,
            "description": "사용자 정의 단계",
            "target_element": "대상 요소",
            "expected_action": "click",
            "category": "카테고리",
            "item_id": "아이템ID",
            "difficulty": "easy"  # easy, medium, hard, very_hard
        }
    ]
}
```

### UI 커스터마이징

`kiosk web/src/` 폴더의 React 컴포넌트를 수정하여 UI를 커스터마이징할 수 있습니다.

## 📈 데이터 분석

### 시선 패턴 분석

```python
import json
import matplotlib.pyplot as plt

# 시선 데이터 로드
with open('gaze_data.json', 'r') as f:
    gaze_data = json.load(f)

# 시선 좌표 추출
x_coords = [data['x'] for data in gaze_data['gaze_data']]
y_coords = [data['y'] for data in gaze_data['gaze_data']]

# 히트맵 생성
plt.hist2d(x_coords, y_coords, bins=50, cmap='hot')
plt.colorbar(label='시선 빈도')
plt.title('시선 패턴 히트맵')
plt.show()
```

### 실패 지점 분석

```python
import pandas as pd

# 실패 데이터 로드
failures_df = pd.read_csv('error_logs.csv')

# 난이도별 실패율 분석
failure_by_difficulty = failures_df.groupby('difficulty').size()
print(failure_by_difficulty)
```

## 🐛 문제 해결

### 일반적인 문제

1. **Eyeware SDK 오류**
   - Eyeware SDK가 설치되지 않은 경우 시선 추적이 비활성화됩니다
   - 시선 추적 없이도 기본 데이터 수집은 가능합니다

2. **React 앱 빌드 실패**
   - Node.js가 설치되어 있는지 확인하세요
   - `npm install`이 성공적으로 실행되었는지 확인하세요

3. **포트 충돌**
   - 다른 프로그램이 같은 포트를 사용하고 있을 수 있습니다
   - 프로그램을 재시작하거나 다른 포트를 사용하세요

### 로그 확인

문제가 발생하면 `collection.log` 파일을 확인하여 상세한 오류 정보를 확인할 수 있습니다.

## 📞 지원

문제가 발생하거나 개선 사항이 있으면 이슈를 등록해주세요.

## 📄 라이선스

이 프로젝트는 연구 목적으로 개발되었습니다. 