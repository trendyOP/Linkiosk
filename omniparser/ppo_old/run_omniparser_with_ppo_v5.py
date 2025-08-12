
# gaze_ppo_saliency_v5.py  (TRAIN + Saliency + Jump‑Action)
# -------------------------------------------------------------
"""
▶ 변경 요약 (v4 → v5)
──────────────────────────────────────────────────────────────
1. **Action space 64개(8×8)** ─ 눈의 saccade(고정점 점프) 모방
   • act ∈ {0..63}  → (gx,gy) 셀 중심으로 즉시 이동
2. **Saliency 보상**
   • 초기화 시 이미지 → 살리언시 맵(8×8) 다운샘플
   • 매 스텝   r += SAL_COEF * H[gy,gx]
3. **KL‑coverage 보너스(선택)**
   • 에피소드 종료 시 방문빈도 vs 히트맵 KL 보상 추가(β)
4. 관찰 벡터 축소
   [goal_prog, pos_x, pos_y, saliency_cell] + BoW(6) + visited(64) = 73
5. 기타
   • no_grad, seed, 주석 동기화, 경로 상수 parameterized
"""

import os, random, numpy as np, cv2, torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Dict
from PIL import Image
from utils.utils import check_ocr_box

# ───── 하이퍼파라미터 ────────────────────────────────────
VISION_GRID_N              = 8
TOTAL_ACTIONS              = VISION_GRID_N * VISION_GRID_N     # 64
MAX_STEPS                  = 50
MOVE_COST                  = -0.005
EXPLORATION_BONUS          = 0.02
DUP_PENALTY                = -0.05
SEMANTIC_REWARD            = 0.2
GOAL_REWARD                = 1.5
EDGE_PENALTY               = -0.07
EDGE_CAP                   = 3
COVERAGE_COEF              = 0.05
COVER_END_THRESH           = 0.9
EMPTY_PENALTY              = -0.15
DENSITY_COEF               = 0.15
TOKEN_DENSITY_MAX          = 10
SAL_COEF                   = 0.07        # ★ saliency reward weight
KL_BETA                    = 0.0         # >0 이면 종료 시 KL‑div 추가 보상

# 디바이스 & 시드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)

# ───── BoW util ─────────────────────────────────────────

def build_vocab_index(vocab: List[str]) -> Dict[str,int]:
    return {t:i for i,t in enumerate(vocab)}

def tokens_to_bow(tokens, vdx):
    v = np.zeros(len(vdx), np.float32)
    for t in tokens:
        idx = vdx.get(t)
        if idx is not None:
            v[idx] = 1.0
    return v

# ───── BBox helpers ─────────────────────────────────────

def bbox_center(bb):
    if isinstance(bb[0], (list, tuple)):
        x = sum(p[0] for p in bb) / 4
        y = sum(p[1] for p in bb) / 4
    else:
        x = (bb[0] + bb[2]) / 2
        y = (bb[1] + bb[3]) / 2
    return x, y


def is_inside(bb, box):
    x, y = bbox_center(bb)
    x1, y1, x2, y2 = box
    return x1 <= x < x2 and y1 <= y < y2

# ───── Saliency util ────────────────────────────────────

def compute_saliency_map(pil_img: Image.Image, N=8):
    """returns NxN numpy array in [0,1]"""
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    try:
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        (ok, salmap) = sal.computeSaliency(img_bgr)
        if not ok:
            raise RuntimeError
        salmap = cv2.resize(salmap.astype(np.float32), (N, N), interpolation=cv2.INTER_AREA)
    except Exception:
        # fallback: uniform map
        salmap = np.ones((N, N), np.float32) / (N * N)
    # normalize 0‑1
    salmap -= salmap.min(); salmap /= salmap.max() + 1e-6
    return salmap

# ───── 환경 ─────────────────────────────────────────────

class GazeKioskEnv:
    def __init__(self, img_path, menu, goal_seq):
        self.image = Image.open(img_path).convert("RGB")
        self.N = VISION_GRID_N
        self.max_steps = MAX_STEPS
        self.menu = menu
        self.vdx = build_vocab_index(menu)

        print("[Env] OCR 수행 중…")
        self.ocr_txt, self.ocr_bb = check_ocr_box(
            self.image, display_img=False, output_bb_format='xyxy', use_paddleocr=True
        )

        print(f"OCR 결과: {len(self.ocr_txt)}개의 텍스트 요소 발견.")
        print("감지된 텍스트:")
        for i, (text, bbox) in enumerate(zip(self.ocr_txt, self.ocr_bb)):
            print(f"  {i+1}. '{text}' at {bbox}")

        # token‑셀 매핑 준비
        self.token_cells = np.zeros((self.N, self.N), bool)
        for bb in self.ocr_bb:
            x, y = bbox_center(bb)
            gx = min(int(x / (1920 / self.N)), self.N - 1)
            gy = min(int(y / (1080 / self.N)), self.N - 1)
            self.token_cells[gy, gx] = True
        self.total_token_cells = self.token_cells.sum().item()

        # saliency heatmap (NxN)
        print("[Env] Saliency map 생성…")
        self.saliency_map = compute_saliency_map(self.image, self.N)
    
        np.set_printoptions(precision=2, suppress=True)
        print("[Env] Saliency grid:\n", self.saliency_map)
        try:
            plt.imshow(self.saliency_map, cmap="hot", interpolation="nearest")
            plt.title("Saliency Grid")
            plt.colorbar()
            plt.savefig("saliency_debug.png")
            plt.close()
            print("    saved saliency_debug.png (grid heatmap)")
        except Exception as e:
            print("    (matplotlib error, skipped heatmap)", e)

        self.set_goal_sequence(goal_seq)
        self.reset()

    # ------------------------------- 환경 API

    def set_goal_sequence(self, seq):
        self.goal_seq = list(seq)

    def reset(self):
        self.goal_idx = self.steps = 0
        self.done = False
        # 시작점: 화면 중앙
        self.pos = [0.5, 0.5]
        self.visited = np.zeros((self.N, self.N), int)
        self.edge_streak = 0
        return self._obs()

    def step(self, a: int):
        # --- 이동 (Jump to cell center)
        gy, gx = divmod(a, self.N)  # 행, 열 순서 고정
        self.pos = [(gx + 0.5) / self.N, (gy + 0.5) / self.N]
        self.steps += 1

        self.visited[gy, gx] += 1
        vcnt = self.visited[gy, gx]
        
        # --- 리워드 계산
        r = MOVE_COST

        # 1) Saliency reward
        # r += SAL_COEF * self.saliency_map[gy, gx]

        # 2) OCR token density
        in_view = [t for t, b in zip(self.ocr_txt, self.ocr_bb) if is_inside(b, self._vbox())]
        token_cnt = len(in_view)
        if token_cnt == 0:
            r += EMPTY_PENALTY
        else:
            density = min(token_cnt, TOKEN_DENSITY_MAX) / TOKEN_DENSITY_MAX
            r += DENSITY_COEF * density

        # 3) exploration / duplicate
        if self.token_cells[gy, gx]:
            if vcnt == 1:
                r += EXPLORATION_BONUS
            else:
                r += DUP_PENALTY / np.sqrt(vcnt)

        # 4) edge penalty
        edge = gx in (0, self.N - 1) or gy in (0, self.N - 1)
        self.edge_streak = self.edge_streak + 1 if edge else 0
        if edge:
            r += EDGE_PENALTY * min(self.edge_streak, EDGE_CAP)

        # 5) coverage bonus
        token_cells_visited = (self.visited & self.token_cells).astype(bool).sum()
        cov = 1 - token_cells_visited / max(1, self.total_token_cells)
        density_scalar = min(token_cnt, TOKEN_DENSITY_MAX) / TOKEN_DENSITY_MAX if token_cnt else 0
        r += COVERAGE_COEF * cov * density_scalar

        # 6) goal reward
        goal = self.goal_seq[self.goal_idx]
        found = False
        if any(goal in t for t in in_view):
            r += SEMANTIC_REWARD + GOAL_REWARD
            self.goal_idx += 1
            found = True

        if self.visited[gy, gx] > 1:
            r += -0.05 * (self.visited[gy, gx]-1)**1.5 

        # --- done 조건
        if token_cells_visited >= self.total_token_cells * COVER_END_THRESH \
           or self.goal_idx >= len(self.goal_seq) \
           or self.steps >= self.max_steps:
            self.done = True
            self.visited[:] = 0  # reset coverage map for 다음 목표
            # 종료 시 KL 보상 (선택)
            if KL_BETA > 0:
                p = (self.visited + 1e-6) / self.visited.sum()
                q = (self.saliency_map + 1e-6) / self.saliency_map.sum()
                r += -KL_BETA * float((p * np.log(p / q)).sum())

        return self._obs(), float(r), self.done, {"found": found}

    # ---- helpers
    def _vbox(self):
        W, H = 1920, 1080
        bw, bh = W / self.N, H / self.N
        cx, cy = int(self.pos[0] * W), int(self.pos[1] * H)
        return max(0, cx - bw / 2), max(0, cy - bh / 2), min(W, cx + bw / 2), min(H, cy + bh / 2)

    def _obs(self):
        in_view = [t for t, b in zip(self.ocr_txt, self.ocr_bb) if is_inside(b, self._vbox())]
        sal_val = self.saliency_map[int(self.pos[1] * self.N), int(self.pos[0] * self.N)]
        bow = tokens_to_bow(in_view, self.vdx)
        return np.concatenate([
            [self.goal_idx / len(self.goal_seq), *self.pos, sal_val],
            bow,
            self.visited.flatten()
        ]).astype(np.float32)

# ───── 모델 ─────────────────────────────────────────────

class GazeActorCritic(nn.Module):
    def __init__(self, obs_dim, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU()
        )
        self.actor = nn.Linear(hid, TOTAL_ACTIONS)
        self.critic = nn.Linear(hid, 1)

    def forward(self, x):
        z = self.net(x)
        return self.actor(z), self.critic(z).squeeze(-1)

# ───── GAE ─────────────────────────────────────────────

def compute_gae(rew, val, g=0.99, lmb=0.95):
    val = val + [0.0]
    adv, gae = [], 0.0
    for t in reversed(range(len(rew))):
        delta = rew[t] + g * val[t + 1] - val[t]
        gae = delta + g * lmb * gae
        adv.insert(0, gae)
    return adv, [a + v for a, v in zip(adv, val[:-1])]

# ───── 학습 루프 ───────────────────────────────────────

def train():
    menu = ['커피', '주문', '아이스크림', '고구마', '티라미수', '메뉴','케이크', 
            '31', '망고케이크', '애니멀파','레디팩','블록팩','듬뿍딸기케이크','음료',
            '파티용품','32000원', '30000원', '치즈', '큐브','골라먹는27', '리얼초코27']
    env = GazeKioskEnv("screen2.png", menu, ["dummy"])

    # 2️⃣ OCR 결과 기반으로 화면에 **실제로 존재하는 토큰(present)** 필터링
    present = [t for t in menu
               if any(t in o for o in env.ocr_txt)]
    print("OCR에 존재하는 토큰:", present)

    # 3️⃣ 이 present만으로 goal 시퀀스를 뽑는 샘플러 정의
    sampler = lambda: random.sample(present, 4)

    
    obs_dim = env.reset().shape[0]

    agent = GazeActorCritic(obs_dim).to(device)
    opt = optim.Adam(agent.parameters(), 2.5e-4)

    ep_r_history, ep_len_history = [], []
    TOTAL_EP = 1500

    for ep in range(TOTAL_EP):
        env.set_goal_sequence(sampler())
        obs = env.reset()
        buf = []
        ep_return = 0.0

        while True:
            logits, _ = agent(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample().item()
            logp = dist.log_prob(torch.tensor(act, device=device)).item()

            nxt, r, done, _ = env.step(act)
            buf.append((obs, act, r, logp))
            obs = nxt; ep_return += r
            if done:
                break

        # --- GAE 계산
        obs_arr, act_arr, rew_arr, logp_arr = list(zip(*buf))
        with torch.no_grad():
            val_arr = [agent(torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0))[1].item() for o in obs_arr]
        adv, ret = compute_gae(rew_arr, val_arr)

        obs_t = torch.tensor(np.array(obs_arr), dtype=torch.float32, device=device)
        acts = torch.tensor(act_arr, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)
        logp_old = torch.tensor(logp_arr, dtype=torch.float32, device=device)

        for _ in range(6):  # PPO epochs
            idx = torch.randperm(len(obs_t))
            for i in range(0, len(idx), 32):
                mb = idx[i:i + 32]
                logits, val = agent(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(acts[mb])
                ratio = torch.exp(logp - logp_old[mb])
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[mb]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = (val - ret_t[mb]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 0.5); opt.step()

        ep_r_history.append(ep_return)
        ep_len_history.append(len(buf))
        if (ep + 1) % 50 == 0:
            print(f"[EP {ep + 1}] avgR={np.mean(ep_r_history[-100:]):.2f}  avgLen={np.mean(ep_len_history[-100:]):.1f}")

    # --- 그래프 저장
    os.makedirs("omniparser/gaze_logs_v5", exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(ep_r_history); plt.title("Episode Reward")
    plt.subplot(1, 2, 2); plt.plot(ep_len_history); plt.title("Episode Length")
    plt.tight_layout(); plt.savefig("omniparser/gaze_logs_v5/learning_curve.png")

    torch.save(agent.state_dict(), "omniparser/gaze_ppo_v5.pt")
    print("saved gaze_ppo_v5.pt  +  learning_curve.png")


if __name__ == "__main__":
    train()