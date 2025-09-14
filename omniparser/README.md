# OmniParser PPO v9 - 보상 체계 및 고령자 시야 구현 가이드

## 📋 목차
1. [개요](#개요)
2. [보상 체계 완전 분석](#보상-체계-완전-분석)
3. [고령자 시야 구현 (Appraisal Module)](#고령자-시야-구현-appraisal-module)
4. [학습 하이퍼파라미터](#학습-하이퍼파라미터)
5. [환경 설정](#환경-설정)
6. [사용법](#사용법)

---

## 개요

OmniParser PPO v9는 키오스크 화면 탐색을 위한 강화학습 에이전트로, 특히 **고령자의 인지적 특성**을 반영한 보상 체계를 구현했습니다. 이 시스템은 9×9 그리드 기반의 시야 탐색을 통해 화면의 모든 영역을 효율적으로 커버하는 것을 목표로 합니다.

---

## 보상 체계 완전 분석

### 1. 기본 보상 설정 (RewardConfig)

```python
@dataclass
class RewardConfig:
    # 셀 방문
    first_cell      : float = +0.35   # 빈칸이라도 첫 방문 (증가)
    revisit_cell    : float = -0.08   # 중복 방문 패널티 완화

    # 텍스트 특별 가중
    first_token_bonus: float = +0.30  # 텍스트 셀 보너스 증가
    dist_shaping     : float = +0.40  # 거리 셰이핑 강화

    # 연속 커버리지 shaping
    step_cov_bonus  : float = +0.15   # 새 셀마다 보너스 증가

    # 에피소드
    max_steps       : int   = 400     # 최대 스텝 증가
    coverage_bonus  : float = +3.0    # 완전 커버리지 보너스 증가

    hint_weight: float = 0.4          # 힌트 가중치 증가
```

### 2. 추가 보상 상수들

```python
# 매직 넘버들을 상수로 정의
SALIENCY_WEIGHT = 0.15                    # 시각적 주목도 가중치 증가
APPRAISAL_PENALTY_WEIGHT = 0.01           # 노인 인지 평가 패널티
LINE_COVERAGE_BONUS = 0.4                 # 라인 커버리지 보너스 증가
COL_COVERAGE_BONUS = 0.4                  # 열 커버리지 보너스 증가
MOVE_DISTANCE_PENALTY = 0.03              # 이동 거리 패널티 완화
DIRECTION_CONTINUITY_BONUS = 0.03         # 방향 연속성 보너스 증가
REVISIT_PENALTY_LIGHT = 0.005             # 중복 방문 경량 패널티 완화
REVISIT_PENALTY_HEAVY = 0.015             # 중복 방문 중량 패널티 완화
COVERAGE_RESET_GRACE_PERIOD = 150         # 유예 기간 연장

# 새로운 커버리지 보상 상수
COVERAGE_PROGRESS_BONUS = 0.05            # 커버리지 진행도 보너스
EDGE_EXPLORATION_BONUS = 0.1              # 가장자리 탐색 보너스
```

### 3. 단계별 보상 계산 과정

#### 3.1 기본 셀 방문 보상
```python
# 기본 방문 보상
r = rc.first_cell if vcnt==1 else rc.revisit_cell  # +0.35 또는 -0.08

# 텍스트 셀 첫 방문 보너스
if is_token and vcnt==1:
    r += rc.first_token_bonus  # +0.30
```

#### 3.2 힌트 맵 보상
```python
r += rc.hint_weight * float(self.hint_map[self.gy, self.gx])  # +0.4 × 힌트값
```

#### 3.3 거리 기반 셰이핑 (Distance Shaping)
```python
# 가장 가까운 미방문 텍스트까지의 거리 기반 보상
unvis_tok = self.token_cells & (self.visited==0)
if unvis_tok.any():
    tgt_gy, tgt_gx = np.column_stack(np.where(unvis_tok))[0]
    dist = abs(tgt_gx-self.gx)+abs(tgt_gy-self.gy)
    r += rc.dist_shaping/(1+dist)  # +0.40/(1+거리)
```

#### 3.4 시각적 주목도 보상
```python
r += SALIENCY_WEIGHT * self.saliency_map[self.gy,self.gx]  # +0.15 × 주목도
```

#### 3.5 커버리지 셰이핑
```python
# 새 셀 방문 시 커버리지 보너스
new_cov = self._grid_cov()
if new_cov > self.prev_cov:
    r += rc.step_cov_bonus  # +0.15
```

#### 3.6 노인 인지 평가 (Appraisal) 보상
```python
# Elderly Appraisal 계산 및 선택적 보상 셰이핑
app_vec = self._compute_appraisal()
if self.appraisal_mode == "RSv1":
    r -= APPRAISAL_PENALTY_WEIGHT * (1.0 - float(app_vec[0]))  # -0.01 × (1-mot_rel)
```

#### 3.7 라인/열 커버리지 보상
```python
# 행 전체 방문 시 보상
if (self.gy not in self.line_visited) and (self.visited[self.gy,:]>0).all():
    r += LINE_COVERAGE_BONUS  # +0.4
    self.line_visited.add(self.gy)

# 열 전체 방문 시 보상  
if (self.gx not in self.col_visited) and (self.visited[:,self.gx]>0).all():
    r += COL_COVERAGE_BONUS  # +0.4
    self.col_visited.add(self.gx)
```

#### 3.8 이동 거리 패널티
```python
# 멀리 점프할수록 패널티
if move_dist > 1:
    r -= MOVE_DISTANCE_PENALTY * (move_dist-1)  # -0.03 × (거리-1)
```

#### 3.9 중복 방문 패널티 (적응적)
```python
# 커버리지 리셋 후에는 패널티 완화
if vcnt > 1:
    if self.steps - self.last_coverage_reset < COVERAGE_RESET_GRACE_PERIOD:
        r -= REVISIT_PENALTY_LIGHT * (vcnt-1)  # -0.005 × (방문횟수-1)
    else:
        r -= REVISIT_PENALTY_HEAVY * (vcnt-1)  # -0.015 × (방문횟수-1)
```

#### 3.10 이동 방향 연속성 보상
```python
# 같은 방향 연속 이동 시 보상
if prev_dir is not None and move_dir == prev_dir and move_dir != (0,0):
    r += DIRECTION_CONTINUITY_BONUS  # +0.03
```

#### 3.11 커버리지 진행도 보상 (새로 추가)
```python
# 커버리지 진행도에 비례한 보상
if new_cov > self.prev_cov:
    r += COVERAGE_PROGRESS_BONUS * (new_cov - self.prev_cov) * 100  # +0.05 × 진행도
```

#### 3.12 가장자리 탐색 보상 (새로 추가)
```python
# 경계 근처 셀 방문 시 보상
edge_distance = min(self.gx, self.gy, self.N-1-self.gx, self.N-1-self.gy)
if edge_distance <= 1 and vcnt == 1:  # 가장자리에서 첫 방문
    r += EDGE_EXPLORATION_BONUS  # +0.1
```

#### 3.13 커버리지 100% 달성 시 특별 처리
```python
# 모든 셀 방문 성공 시
if new_cov >= 1.0:
    r += rc.coverage_bonus  # +3.0
    # 커버리지 초기화하여 재탐색 유도
    self.visited.fill(0)
    self.prev_cov = 0.0
    self.line_visited.clear()
    self.col_visited.clear()
    self.last_coverage_reset = self.steps
    # 새로운 시작점으로 랜덤 이동
    self.gx, self.gy = np.random.randint(0, self.N, size=2)
    self.pos = [(self.gx+0.5)/self.N, (self.gy+0.5)/self.N]
```

#### 3.14 에피소드 종료 패널티
```python
# 최대 스텝 초과 시 패널티
elif self.steps >= rc.max_steps:
    r -= 1.0; done = True
```

---

## 고령자 시야 구현 (Appraisal Module)

### 1. AppraisalModuleElderlyV2 클래스

고령자의 인지적 특성을 반영한 6차원 평가 벡터를 생성합니다.

```python
class AppraisalModuleElderlyV2:
    HIGH_CONTRAST_COLORS = {"red", "yellow", "white", "빨강", "노랑"}
    FOVEAL_DEG = 10
    CRT_SLOPE_MS_PER_YEAR = 0.0028
    PROPRIO_ERROR_THRESH_CM = 3.0

    def __init__(self, env: "GazeKioskEnv"):
        self.env = env
        self.base_cert = (0.25, 0.55)
```

### 2. 6차원 평가 벡터 구성

#### 2.1 Motivational Relevance (동기적 관련성)
```python
# 1. motivational relevance
mot_rel = 0.0
if grid_idx:
    vx, vy, vw, vh = self.env._grid_to_viewport(*grid_idx)
    cx, cy = vx + vw / 2, vy + vh / 2
    cdist = math.hypot(cx - self.env.W / 2, cy - self.env.H / 2)
    fov_r = math.tan(math.radians(self.FOVEAL_DEG)) * self.env.H / 2
    mot_rel = 1.0 - min(1.0, cdist / max(fov_r, 1.0))
    if any(tok.lower() in self.HIGH_CONTRAST_COLORS for tok in in_view_tokens):
        mot_rel = min(1.0, mot_rel + 0.15)
```

**구현 원리:**
- 시야 중심으로부터의 거리 기반 계산
- 고대비 색상(빨강, 노랑, 흰색)에 대한 선호도 반영
- 중심 시야(10도) 내의 요소에 높은 관련성 부여

#### 2.2 Certainty (확실성)
```python
# 2. certainty
cert_low, cert_hi = self.base_cert
crt_penalty = max(0.0, (age_years - 60) * self.CRT_SLOPE_MS_PER_YEAR)
certainty = np.clip(random.uniform(cert_low, cert_hi) - crt_penalty, 0, 1)
```

**구현 원리:**
- 나이에 따른 반응 시간 증가 반영 (CRT_SLOPE_MS_PER_YEAR = 0.0028)
- 60세 이후부터 확실성이 점진적으로 감소
- 기본 확실성 범위: 0.25 ~ 0.55

#### 2.3 Novelty (새로움)
```python
# 3. novelty
novelty = max(0.0, 1.0 - visited_ratio) * 0.5
```

**구현 원리:**
- 미방문 영역에 대한 호기심 반영
- 방문 비율이 높을수록 새로움 감소
- 최대 0.5까지 제한

#### 2.4 Goal Congruence (목표 일치성)
```python
# 4. goal congruence
goal_cong = 1.0 if mot_rel > 0.8 else 0.0
```

**구현 원리:**
- 동기적 관련성이 높을 때 목표 일치성 증가
- 임계값 0.8 이상에서만 목표 일치로 판단

#### 2.5 Coping Potential (대처 능력)
```python
# 5. coping potential
coping = 0.3 + 0.4 * visited_ratio
if proprio_endpt_err_cm > self.PROPRIO_ERROR_THRESH_CM:
    coping = max(0.0, coping - 0.2)
```

**구현 원리:**
- 방문 경험에 따른 대처 능력 증가
- 고유수용성 오류가 3cm 이상일 때 대처 능력 감소
- 기본 대처 능력: 0.3 ~ 0.7

#### 2.6 Anticipation (기대감)
```python
# 6. anticipation
anticipation = (1 - certainty) * (0.8 + novelty / 2)
```

**구현 원리:**
- 확실성이 낮을수록 기대감 증가
- 새로움과 결합하여 동기 부여
- 불확실성과 호기심의 상호작용 모델링

### 3. 환경 변수 설정

```python
# 환경 변수로 고령자 특성 조정 가능
self.appraisal_age = int(os.environ.get("APPRAISAL_AGE", "70"))  # 나이
self.proprio_endpt_err_cm = float(os.environ.get("APPRAISAL_PROPRIO_ERR_CM", "4.0"))  # 고유수용성 오류
self.appraisal_mode = os.environ.get("APPRAISAL_MODE", "RSv1")  # 평가 모드
```

### 4. 관찰 벡터에 통합

```python
def _obs(self, app_vec=None):
    # ... 기존 관찰 벡터 구성 ...
    
    # [APPRAISAL-ADD] Elderly Appraisal 6D 벡터 추가
    if app_vec is None:
        app_vec = self._compute_appraisal()
    app_vec = app_vec.astype(np.float32)

    vec = np.concatenate([
        self.pos,                   # 2 (위치)
        [self._grid_cov()],         # 1 (커버리지)
        dir_vec,                    # 2 (방향 벡터)
        bow,                        # |V| (어휘 벡터)
        visited_bin,                # N*N (방문 마스크)
        hint_flat,                  # N*N (힌트 맵)
        app_vec                     # 6 (고령자 평가 벡터)
    ])
    return vec.astype(np.float32)
```

---

## 학습 하이퍼파라미터

```python
@dataclass
class HP:
    lr               : float = 3.0e-4    # 학습률 약간 증가
    batch_size       : int = 512
    epochs_per_update: int = 6           # 업데이트 에포크 증가
    gamma            : float = 0.99
    lam              : float = 0.95
    total_episodes   : int = 4000        # 총 에피소드 수 증가
    ent_coef         : float = 0.10      # 엔트로피 계수 증가 (탐험 강화)
```

### 하이퍼파라미터 설명

- **lr (학습률)**: 3.0e-4 - 안정적인 학습을 위한 적절한 학습률
- **batch_size**: 512 - 메모리 효율성과 학습 안정성의 균형
- **epochs_per_update**: 6 - PPO 업데이트 시 충분한 학습 반복
- **gamma**: 0.99 - 장기적 보상에 대한 높은 할인율
- **lam**: 0.95 - GAE(Generalized Advantage Estimation) 람다 값
- **total_episodes**: 4000 - 충분한 학습을 위한 에피소드 수
- **ent_coef**: 0.10 - 탐험을 강화하는 엔트로피 계수

---

## 환경 설정

### 1. 기본 설정

```python
# Constants & Config
CONFIG_PATH = "omniparser/config.ini"
DEFAULT_IMAGE_PATH = "screen5.png"
LOG_DIR = "runs/gaze_scan_v2"
MODEL_SAVE_PATH = "omniparser/gaze_ppo_v9.pt"
LOGS_DIR = "omniparser/gaze_logs_v9"

# 그리드 설정
VISION_GRID_N: int = 9  # 9×9 그리드
MOVE_OPTIONS: Tuple[Tuple[int,int], ...] = (
    (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)
)  # 8방향 이동
TOTAL_ACTIONS = len(MOVE_OPTIONS)  # 8개 액션
```

### 2. 설정 파일 (config.ini)

```ini
[User]
type = elder
vision_grid = 9

[Menu]
queue = 매장식사, 아메리카노, 주문담기, 더담기, 카푸치노, 주문담기,결제하기, 확인, 신용카드, 대기, 아니오
```

### 3. 시드 설정 (재현성)

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## 사용법

### 1. 학습 실행

```bash
cd omniparser
python run_omniparser_with_ppo_v9.py
```

### 2. 학습된 모델 테스트

```bash
python policy_demo.py
```

### 3. 환경 변수 설정 (선택사항)

```bash
# 고령자 나이 설정
export APPRAISAL_AGE=75

# 고유수용성 오류 설정 (cm)
export APPRAISAL_PROPRIO_ERR_CM=5.0

# 평가 모드 설정
export APPRAISAL_MODE=RSv1
```

### 4. TensorBoard로 학습 모니터링

```bash
tensorboard --logdir=runs/gaze_scan_v2
```

---

## 보상 체계의 특징

### 1. 다층적 보상 구조
- **기본 보상**: 셀 방문 (+0.35/-0.08)
- **텍스트 보상**: 텍스트 셀 특별 가중 (+0.30)
- **거리 보상**: 목표까지의 거리 기반 셰이핑
- **커버리지 보상**: 전체적인 탐색 진행도
- **행동 보상**: 연속성, 효율성 등

### 2. 적응적 패널티 시스템
- 커버리지 리셋 후 150스텝 동안 중복 방문 패널티 완화
- 이동 거리에 따른 점진적 패널티
- 노인 인지 특성을 고려한 평가 시스템

### 3. 셰이핑 기법
- **Distance Shaping**: 목표까지의 거리 기반 보상
- **Coverage Shaping**: 전체 커버리지 진행도 기반 보상
- **Direction Shaping**: 이동 방향 연속성 기반 보상
- **Edge Shaping**: 가장자리 탐색 유도 보상

### 4. 노인 친화적 설계
- **Appraisal Module**: 노인의 인지적 특성 반영
- **Motivational Relevance**: 시야 중심 거리 기반 동기 부여
- **Certainty**: 나이에 따른 반응 시간 고려
- **Novelty**: 미방문 영역에 대한 호기심 반영
- **Coping Potential**: 경험과 고유수용성 오류 고려
- **Anticipation**: 불확실성과 기대감의 상호작용

---

## 파일 구조

```
omniparser/
├── run_omniparser_with_ppo_v9.py    # 메인 학습 스크립트
├── policy_demo.py                   # 학습된 모델 테스트
├── config.ini                       # 환경 설정 파일
├── gaze_ppo_v9.pt                   # 학습된 모델 가중치
├── README.md                        # 이 문서
└── gaze_logs_v9/                    # 학습 로그 디렉토리
```

---

## 참고사항

1. **GPU 사용**: CUDA가 사용 가능한 경우 자동으로 GPU를 사용합니다.
2. **메모리 요구사항**: 배치 크기 512에 따라 충분한 GPU 메모리가 필요합니다.
3. **학습 시간**: 4000 에피소드 완료까지 GPU 환경에서 약 2-4시간 소요됩니다.
4. **모니터링**: TensorBoard를 통해 실시간 학습 진행 상황을 확인할 수 있습니다.

---

*이 문서는 OmniParser PPO v9의 보상 체계와 고령자 시야 구현에 대한 완전한 가이드입니다.*
