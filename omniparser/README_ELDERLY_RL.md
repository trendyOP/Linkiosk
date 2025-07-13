# 고령자 강화학습 에이전트 시스템

이 시스템은 수집된 UIUX 인증 데이터를 사용하여 고령자 특성을 반영한 강화학습 에이전트를 훈련하고 실행합니다.

## 🎯 주요 특징

### 고령자 특성 반영
- **기억력 저하 시뮬레이션**: 최근 본 정보를 잊어버리는 메모리 시스템
- **느린 반응 속도**: 천천히 움직이는 것에 보상, 빠른 행동에 패널티
- **시선 고정 시간**: 한 곳에 오래 머무르는 자연스러운 행동
- **반복 탐색**: 같은 위치를 반복 탐색하는 현실적인 패턴

### 강화학습 구성요소
- **Actor-Critic 네트워크**: 정책과 가치 함수를 동시에 학습
- **고령자 메모리 시스템**: 기억력 저하를 시뮬레이션
- **특화된 보상 함수**: 고령자 특성을 반영한 보상 설계

## 📁 파일 구조

```
omniparser/
├── reinforcement_learning_agent.py    # 고령자 RL 에이전트 클래스
├── train_elderly_rl.py              # 훈련 스크립트
├── run_omniparser_with_elderly_rl.py # 훈련된 에이전트 실행
└── README_ELDERLY_RL.md             # 이 파일
```

## 🚀 사용 방법

### 1. 훈련 데이터 준비

먼저 UIUX 인증 데이터가 수집되어 있어야 합니다:
```
beameyetracker/kiosk_data/
├── uiux_certification_20250712_180612/
│   ├── scenario_data.json
│   ├── gaze_data.json
│   ├── click_events.json
│   ├── timeline_data.json
│   └── performance_metrics.json
└── ...
```

### 2. 강화학습 에이전트 훈련

```bash
cd omniparser
python train_elderly_rl.py --data_dir ../beameyetracker/kiosk_data --num_episodes 1000
```

#### 훈련 옵션
- `--data_dir`: 훈련 데이터 디렉토리 경로
- `--num_episodes`: 훈련 에피소드 수 (기본값: 1000)
- `--batch_size`: 배치 크기 (기본값: 32)
- `--learning_rate`: 학습률 (기본값: 0.001)
- `--output_dir`: 훈련된 모델 저장 디렉토리 (기본값: trained_models)

### 3. 훈련된 에이전트 실행

```bash
python run_omniparser_with_elderly_rl.py
```

## 🧠 고령자 특성 상세 설명

### 1. 기억력 저하 (ElderlyMemory)

```python
class ElderlyMemory:
    def __init__(self, max_items=3, forget_steps=5, memory_decay_rate=0.8):
        # max_items: 한 번에 기억할 수 있는 최대 아이템 수
        # forget_steps: 몇 스텝 후에 잊어버릴지
        # memory_decay_rate: 기억 신뢰도 감소율
```

**특징:**
- 최대 3개의 UI 요소만 동시에 기억
- 5스텝 후 자동으로 잊어버림
- 기억 신뢰도가 시간이 지나면서 감소

### 2. 보상 함수 설계

```python
def calculate_elderly_reward(self, action, detected_buttons, target_menu, 
                            stamp_position, reaction_time, gaze_velocity):
```

**보상 구성요소:**
- **목표 찾기**: 목표 메뉴를 찾으면 +10점
- **메모리 활용**: 기억에 있는 정보를 활용하면 +5점
- **반응 속도**: 2초 이상 천천히 반응하면 +1점, 0.5초 이하 빠르면 -1점
- **시선 속도**: 너무 빠른 시선 이동에 패널티
- **반복 탐색**: 같은 행동 반복 시 -3점
- **거리 보상**: 목표와 가까우면 +2점, 멀면 -1점

### 3. 상태 벡터 구성 (150차원)

```python
state_vector = [
    # 1. 시선 위치 (2차원)
    x_norm, y_norm,
    
    # 2. 시선 신뢰도 (1차원)
    confidence,
    
    # 3. 시나리오 진행률 (1차원)
    progress,
    
    # 4. 시선 속도 (1차원)
    velocity,
    
    # 5. 시선 가속도 (1차원)
    acceleration,
    
    # 6. 최근 클릭과의 시간 차이 (1차원)
    time_diff_norm,
    
    # 7. 시선 고정 시간 (1차원)
    fixation_duration,
    
    # 8. 시나리오 단계별 정보 (10차원)
    step_completion_status,
    
    # 9. 메모리 상태 (2차원)
    memory_items, memory_confidence,
    
    # 10. 패딩 (131차원)
    padding...
]
```

## 📊 훈련 결과 분석

### 훈련 진행 상황 시각화

훈련 후 다음 파일들이 생성됩니다:
- `training_progress_YYYYMMDD_HHMMSS.png`: 훈련 진행 상황 그래프
- `elderly_rl_agent_YYYYMMDD_HHMMSS.pth`: 훈련된 모델

### 그래프 해석

1. **Episode Rewards**: 에피소드별 보상 변화
2. **Actor Loss**: 정책 네트워크 손실
3. **Critic Loss**: 가치 네트워크 손실
4. **Entropy**: 행동 다양성 (높을수록 탐색적)

## 🔧 고급 설정

### 메모리 파라미터 조정

```python
# 더 심한 기억력 저하
memory = ElderlyMemory(max_items=2, forget_steps=3, memory_decay_rate=0.7)

# 더 가벼운 기억력 저하
memory = ElderlyMemory(max_items=5, forget_steps=10, memory_decay_rate=0.9)
```

### 보상 함수 커스터마이징

```python
def custom_elderly_reward(self, action, detected_buttons, target_menu, 
                          stamp_position, reaction_time, gaze_velocity):
    reward = 0.0
    
    # 고령자 특성에 맞게 보상 조정
    if reaction_time > 3.0:  # 더 느린 반응에 보상
        reward += 2.0
    
    if gaze_velocity < 50:  # 더 느린 시선 이동에 보상
        reward += 1.5
    
    return reward
```

## 🎮 실행 예시

### 훈련 실행
```bash
# 기본 훈련
python train_elderly_rl.py

# 고급 훈련 (더 많은 에피소드, 작은 배치)
python train_elderly_rl.py --num_episodes 2000 --batch_size 16 --learning_rate 0.0005

# 특정 데이터 디렉토리 사용
python train_elderly_rl.py --data_dir ../custom_data --output_dir ../custom_models
```

### 훈련된 에이전트 실행
```bash
# 훈련된 모델이 있으면 자동으로 로드
python run_omniparser_with_elderly_rl.py
```

## 📈 성능 평가

### 고령자 특성 지표

1. **기억 활용률**: 기억에 있는 정보를 활용한 성공률
2. **반응 시간**: 평균 반응 시간 (2-3초가 적정)
3. **시선 이동 속도**: 평균 시선 이동 속도 (100-500px/s가 적정)
4. **반복 탐색률**: 같은 위치를 반복 탐색하는 비율

### 성공률 측정

```python
# 훈련 후 성능 평가
final_rewards = training_history['episode_rewards'][-100:]
avg_reward = np.mean(final_rewards)
print(f"최종 평균 보상: {avg_reward:.3f}")
```

## 🐛 문제 해결

### 일반적인 문제들

1. **CUDA 메모리 부족**
   ```bash
   # CPU만 사용
   export CUDA_VISIBLE_DEVICES=""
   python train_elderly_rl.py
   ```

2. **데이터 로드 실패**
   ```bash
   # 데이터 경로 확인
   ls -la beameyetracker/kiosk_data/
   ```

3. **훈련이 수렴하지 않음**
   ```bash
   # 학습률 낮추기
   python train_elderly_rl.py --learning_rate 0.0001
   ```

## 🔮 향후 발전 방향

1. **다양한 고령자 특성 모델링**
   - 시각 장애 시뮬레이션
   - 운동 기능 저하 모델링
   - 인지 능력 차이 반영

2. **실시간 적응 시스템**
   - 사용자별 특성 학습
   - 실시간 보상 조정
   - 개인화된 도움말 제공

3. **멀티모달 학습**
   - 음성 명령 인식
   - 제스처 인식
   - 감정 상태 반영

## 📚 참고 자료

- [Reinforcement Learning with Human Feedback](https://arxiv.org/abs/1706.03741)
- [Elderly User Interface Design Guidelines](https://www.w3.org/WAI/older-users/)
- [Eye Tracking in Human-Computer Interaction](https://doi.org/10.1016/j.ijhcs.2008.09.008)

---

**주의**: 이 시스템은 고령자 특성을 시뮬레이션하지만, 실제 고령자와는 차이가 있을 수 있습니다. 실제 적용 시에는 고령자 사용자와의 테스트를 통해 검증이 필요합니다. 