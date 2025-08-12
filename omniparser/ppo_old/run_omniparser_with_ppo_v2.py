# gaze_ppo_refactor_v2_full.py
# -------------------------------------------------------
"""
통합본 (2025-06-17)
──────────────────────────────────────────────────────────
🆕 변경 요약
1. Coverage 보너스            – 미방문 비율 × COVERAGE_BONUS_COEFF
2. Coverage-기반 Early-Stop  – 90 % 이상 커버 시 episode 종료
3. Edge 패널티 / 디버그 로그 – EDGE_PENALTY 값으로 on/off 가능
4. MLP Actor-Critic + PPO   – LSTM 불일치 제거, 간결·안정
5. 고령자 페르소나            – 지연·오입력·오클릭 시뮬레이션
"""
import os, random, time, math
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

# ==== OCR util (사용자 환경에 맞게 수정) ===========================
from utils.utils import check_ocr_box  # PaddleOCR 래퍼

# -------------------------------------------------------------------
# 0. 하이퍼파라미터 / 상수
VISION_GRID_N            = 7
STEP_SIZE                = 1 / VISION_GRID_N
TOTAL_ACTIONS            = 4          # 상, 하, 좌, 우
MAX_STEPS                = 50         # 한 episode 최대 step 수
MOVE_COST                = -0.005
DUP_PENALTY              = -0.05
EXPLORATION_BONUS        = 0.02
SEMANTIC_REWARD          = 0.2
GOAL_REWARD              = 1.5
EDGE_PENALTY             = -0.05      # 0 으로 끄면 비활성
COVERAGE_BONUS_COEFF     = 0.05       # 미방문 비율 × 계수
COVERAGE_THRESHOLD       = 0.9        # 90 % 커버 시 episode 종료
SEED                     = 42         # 재현성

# -------------------------------------------------------------------
#  재현성 → random / numpy / torch seed 고정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# 1. 보조 함수
def build_vocab_index(vocab: List[str]) -> Dict[str, int]:
    """문자열 vocab → {token: index} dict"""
    return {tok: i for i, tok in enumerate(vocab)}

def tokens_to_bow(tokens: List[str], vocab_idx: Dict[str, int]) -> np.ndarray:
    """토큰 리스트를 Bag-of-Words 벡터로 (O(|tokens|))"""
    vec = np.zeros(len(vocab_idx), dtype=np.float32)
    for t in tokens:
        idx = vocab_idx.get(t)
        if idx is not None:
            vec[idx] = 1.0
    return vec

def bbox_center(bbox):
    """bbox(xyxy or 4-pt) → 중심 (x,y)"""
    if isinstance(bbox[0], (list, tuple)):
        x = sum(pt[0] for pt in bbox) / 4
        y = sum(pt[1] for pt in bbox) / 4
    else:
        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2
    return x, y

def is_inside(bbox, box_xyxy):
    x, y = bbox_center(bbox)
    x1, y1, x2, y2 = box_xyxy
    return x1 <= x < x2 and y1 <= y < y2

# -------------------------------------------------------------------
# 2. 환경
class GazeKioskEnv:
    """OCR 기반 시야 박스 이동 환경 (coverage + edge + debug)"""

    def __init__(
        self,
        image_path: str,
        full_menu_list: List[str],
        initial_goal_sequence: List[str],
        screen_size: Tuple[int, int] = (1920, 1080),
        vision_grid_n: int = VISION_GRID_N,
        max_steps: int = MAX_STEPS,
        debug: bool = False,
    ):
        self.image = Image.open(image_path)
        self.screen_size = screen_size
        self.N = vision_grid_n
        self.max_steps = max_steps
        self.step_size = 1 / vision_grid_n
        self.full_menu = full_menu_list
        self.vocab_idx = build_vocab_index(full_menu_list)
        self.debug = debug

        print(f"\n[Env] 이미지 처리: {image_path}")
        print("[Env] OCR 수행 중…")
        self.ocr_text, self.ocr_bbox = check_ocr_box(
            self.image,
            display_img=False,
            output_bb_format="xyxy",
            goal_filtering=None,
            use_paddleocr=True,
        )
        print(f"[Env] OCR 결과: {len(self.ocr_text)}개 텍스트 요소 발견.")

        self.set_goal_sequence(initial_goal_sequence)
        self.reset()

    # ---------------------------------------------------
    def set_goal_sequence(self, seq: List[str]):
        self.goal_sequence = seq

    def reset(self):
        self.goal_idx  = 0
        self.steps     = 0
        self.done      = False
        self.pos       = [0.5, 0.5]                       # 0-1 정규화 중심
        self.visited   = np.zeros((self.N, self.N), np.int32)
        return self._get_obs()

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Call reset() before step()")

        # ---------------- 이동 (상하좌우)
        dx, dy = (
            (0, -self.step_size) if action == 0 else
            (0,  self.step_size) if action == 1 else
            (-self.step_size, 0) if action == 2 else
            ( self.step_size, 0)
        )
        self.pos[0] = np.clip(self.pos[0] + dx, 0, 1)
        self.pos[1] = np.clip(self.pos[1] + dy, 0, 1)
        self.steps += 1

        reward = MOVE_COST

        # ---------------- Grid 인덱스
        gy = min(int(self.pos[1] * self.N), self.N - 1)
        gx = min(int(self.pos[0] * self.N), self.N - 1)
        self.visited[gy, gx] += 1
        v_cnt = self.visited[gy, gx]
        reward += EXPLORATION_BONUS if v_cnt == 1 else DUP_PENALTY / math.sqrt(v_cnt)

        # ---------------- Edge 패널티
        on_edge = gx in (0, self.N - 1) or gy in (0, self.N - 1)
        if on_edge:
            reward += EDGE_PENALTY

        # ---------------- Coverage 보너스
        visited_cnt = (self.visited > 0).sum()
        coverage_ratio = 1.0 - visited_cnt / (self.N * self.N)  # 미방문 비율
        reward += COVERAGE_BONUS_COEFF * coverage_ratio

        # ---------------- OCR 검사 & 목표 처리
        in_view = [txt for txt, bb in zip(self.ocr_text, self.ocr_bbox)
                   if is_inside(bb, self._vision_box_xyxy())]
        current_goal = self.goal_sequence[self.goal_idx]
        found = False
        if any(current_goal in txt for txt in in_view):
            reward += SEMANTIC_REWARD + GOAL_REWARD
            self.goal_idx += 1
            self.visited[:] = 0                           # 새 목표 → 방문 초기화
            found = True

        # ---------------- 종료 조건
        if (
            visited_cnt >= self.N * self.N * COVERAGE_THRESHOLD or
            self.goal_idx >= len(self.goal_sequence) or
            self.steps >= self.max_steps
        ):
            self.done = True

        # ---------------- 디버그 로그
    
        print(
            f"[Step {self.steps}] grid=({gy},{gx}) edge={on_edge} cov={coverage_ratio:.2f} "
            f"goal='{current_goal}' found={found} in_view={in_view}"
        )

        return self._get_obs(), reward, self.done, {"found": found}

    # ---------------------------------------------------
    def _vision_box_xyxy(self):
        W, H = self.screen_size
        box_w, box_h = W / self.N, H / self.N
        cx, cy = int(self.pos[0] * W), int(self.pos[1] * H)
        return (
            max(0, cx - box_w / 2),
            max(0, cy - box_h / 2),
            min(W, cx + box_w / 2),
            min(H, cy + box_h / 2),
        )

    def _get_obs(self) -> np.ndarray:
        in_view = [
            txt
            for txt, bb in zip(self.ocr_text, self.ocr_bbox)
            if is_inside(bb, self._vision_box_xyxy())
        ]
        bow = tokens_to_bow(in_view, self.vocab_idx)
        return np.concatenate([
            np.array([
                self.goal_idx / max(len(self.goal_sequence), 1),
                self.pos[0],
                self.pos[1],
            ], np.float32),
            bow,
            self.visited.astype(np.float32).flatten(),
        ])

# -------------------------------------------------------------------
# 3. 고령자 페르소나 래퍼
class ElderlyPersonaWrapper:
    def __init__(self, env, error_prob=0.3, delay_mu=0.4, delay_sigma=0.1):
        self.env = env
        self.err_p = error_prob
        self.delay_mu = delay_mu
        self.delay_sigma = delay_sigma

    def reset(self):
        return self.env.reset()

    def step(self, action):
        time.sleep(max(0, random.gauss(self.delay_mu, self.delay_sigma)))
        if random.random() < self.err_p:
            action = random.randrange(TOTAL_ACTIONS)
        return self.env.step(action)

# -------------------------------------------------------------------
# 4. MLP Actor-Critic
class GazeActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, TOTAL_ACTIONS)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        z = self.net(x)
        return self.actor(z), self.critic(z).squeeze(-1)

# -------------------------------------------------------------------
# 5. PPO 학습 함수
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages, gae = [], 0.0
    values = values + [0.0]  # bootstrap
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return advantages, returns

def train_ppo(
    image_path: str,
    menu_list: List[str],
    total_episodes=300,
    clip_eps=0.2,
    lr=2.5e-4,
    batch_size=64,
    ppo_epochs=4,
    save_path="gaze_ppo.pt",
    log_dir="logs",
):
    os.makedirs(log_dir, exist_ok=True)

    # 목표 샘플러 (길이 2~4)
    def sample_goal():
        k = random.randint(2, min(4, len(menu_list)))
        return random.sample(menu_list, k=k)

    env = GazeKioskEnv(
        image_path=image_path,
        full_menu_list=menu_list,
        initial_goal_sequence=sample_goal(),
    )
    env = ElderlyPersonaWrapper(env)
    obs_dim = env.reset().shape[0]

    agent = GazeActorCritic(obs_dim).to(device)
    opt = optim.Adam(agent.parameters(), lr=lr)

    ep_rewards, ep_lens = [], []
    for ep in range(total_episodes):
        env.env.set_goal_sequence(sample_goal())
        obs = env.reset()

        traj_obs, traj_act, traj_val, traj_rew, traj_logp = [], [], [], [], []
        done, ep_r, ep_len = False, 0.0, 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = agent(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample().item()
            logp = dist.log_prob(torch.tensor(action, device=device)).item()

            next_obs, reward, done, _ = env.step(action)

            traj_obs.append(obs)
            traj_act.append(action)
            traj_val.append(value.item())
            traj_rew.append(reward)
            traj_logp.append(logp)

            obs = next_obs
            ep_r += reward
            ep_len += 1

        # --- GAE
        adv, ret = compute_gae(traj_rew, traj_val)
        adv = torch.tensor(adv, dtype=torch.float32, device=device)
        ret = torch.tensor(ret, dtype=torch.float32, device=device)
        acts = torch.tensor(traj_act, dtype=torch.int64, device=device)
        obs_batch = torch.tensor(np.array(traj_obs), dtype=torch.float32, device=device)
        old_logp = torch.tensor(traj_logp, dtype=torch.float32, device=device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # --- PPO update
        for _ in range(ppo_epochs):
            idx = torch.randperm(len(obs_batch))
            for start in range(0, len(idx), batch_size):
                mb_idx = idx[start:start + batch_size]
                mb_obs = obs_batch[mb_idx]
                mb_act = acts[mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret = ret[mb_idx]
                mb_old = old_logp[mb_idx]

                logits, values = agent(mb_obs)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(mb_act)

                ratio = torch.exp(logp - mb_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (values - mb_ret).pow(2).mean()
                entropy = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()

        ep_rewards.append(ep_r)
        ep_lens.append(ep_len)
        print(f"[EP {ep+1:4d}] reward {np.mean(ep_rewards[-20:]):.3f} | len {np.mean(ep_lens[-20:]):.1f}")

    # --- 저장 & 로그
    torch.save(agent.state_dict(), save_path)
    print("model saved:", save_path)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(ep_rewards); plt.title("Episode Reward")
    plt.subplot(1, 2, 2); plt.plot(ep_lens);    plt.title("Episode Length")
    plt.tight_layout(); plt.savefig(os.path.join(log_dir, "learning_curve.png"))

# -------------------------------------------------------------------
if __name__ == "__main__":
    menu_list = ["결제", "메뉴", "상품", "진정한티라미수", "스노우볼", "커피"]
    train_ppo(
        image_path="screen2.png",
        menu_list=menu_list,
        total_episodes=300,
        save_path="omniparser/gaze_ppo_v3.pt",
        log_dir="omniparser/gaze_logs_v3",
    )
