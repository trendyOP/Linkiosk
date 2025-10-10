# 마우스 속도 프로필 설정

## 개요
이 문서는 OmniParser에서 사용되는 사용자별 마우스 속도 프로필 설정을 정리한 것입니다.

## 프로필 설정

### 1. 고령자 (Elder) 프로필
```python
@dataclass
class ElderProfile:
    # 마우스 움직임 (운동)
    mouse_speed_px_s: float = 800   # 고령자: 느린 속도 (800 px/s)
    path_curvature: float = 0.08     # 경로 곡률
    tremor_std_px: float = 2.0       # 떨림 표준편차 (픽셀)
    overshoot_prob: float = 0.15     # 오버슈트 확률
    overshoot_ratio: float = 0.03    # 오버슈트 비율
```

**특징:**
- 마우스 속도: **800 px/s**
- 곡선 경로 이동 사용
- 떨림 현상 시뮬레이션
- 오버슈트 현상 포함

### 2. 젊은이 (Young) 프로필
```python
@dataclass
class YoungProfile:
    # 마우스 움직임 (운동)
    mouse_speed_px_s: float = 3000   # 젊은이: 빠른 속도 (3000 px/s)
    path_curvature: float = 0.05     # 경로 곡률
    tremor_std_px: float = 0.5       # 떨림 표준편차 (픽셀)
    overshoot_prob: float = 0.05     # 오버슈트 확률
    overshoot_ratio: float = 0.02    # 오버슈트 비율
```

**특징:**
- 마우스 속도: **3000 px/s**
- 직선 경로 이동
- 최소한의 떨림
- 낮은 오버슈트 확률

## 속도 계산 방식

### 젊은이 (직선 이동)
```python
duration = max(0.05, min(0.5, dist / user_behavior['mouse_speed_px_s']))
```
- 최소 지속시간: 0.05초
- 최대 지속시간: 0.5초
- 거리/속도로 계산된 지속시간을 위 범위로 제한

### 고령자 (곡선 이동)
```python
duration = dist / max(80, behavior['mouse_speed_px_s'])
```
- 최소 속도 제한: 80 px/s
- 베지어 곡선을 이용한 자연스러운 경로
- 떨림과 오버슈트 현상 포함

## 주의사항

⚠️ **현재 코드에서 주석과 실제 값이 일치하지 않습니다:**
- 주석: "고령자: 느린 속도 (800 px/s)", "젊은이: 빠른 속도 (3000 px/s)"
- 실제 값: 고령자 8000 px/s, 젊은이 3000 px/s

실제로는 **고령자가 더 빠른 속도**를 가지도록 설정되어 있습니다.

## 사용 예시

```python
# 사용자 행동 프로파일 로드
user_behavior = get_user_behavior(CONFIG['user_type'])

# 마우스 이동 시 속도 적용
if user_behavior['enable_curved_movement']:
    move_mouse_curved(x, y, user_behavior, record_mouse_move)  # 고령자: 곡선 이동
else:
    # 젊은이: 직선 이동 (속도 적용)
    duration = max(0.05, min(0.5, dist / user_behavior['mouse_speed_px_s']))
    pyautogui.moveTo(x, y, duration=duration)
```

## 디버그 출력

실행 시 다음과 같은 디버그 정보가 출력됩니다:

```
[MOUSE SPEED] Elder: 8000 px/s, dist: 150.5px, duration: 0.019s
[MOUSE SPEED] Young: 3000 px/s, dist: 150.5px, duration: 0.050s
[CLICK SPEED] Elder: 8000 px/s, dist: 75.2px, duration: 0.009s
[CLICK SPEED] Young: 3000 px/s, dist: 75.2px, duration: 0.025s
```

## 파일 위치
- 설정 파일: `omniparser/gaze_ppo_test_v9_3.py`
- 라인 번호: 1858-1888 (프로필 정의), 1508, 1543, 1958 (속도 적용)



