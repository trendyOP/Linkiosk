
from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────────
# Imports & Globals
# ────────────────────────────────────────────────────────────────────────────────
import os, random, math, configparser, itertools, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from text_normalizer import normalize_token

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt 
from torch.utils.tensorboard import SummaryWriter  # ← 변경

from utils.utils import check_ocr_box  # 내부 유틸 그대로 사용
import re


# matplotlib 백엔드 설정은 utils 모듈 내로 이동 (시각화 함수만 사용할 때 로드)

# ────────────────────────────────────────────────────────────────────────────────
# Config & Hyper‑parameters
# ────────────────────────────────────────────────────────────────────────────────

SEED = 42
CURRENCY_TAIL_RE = re.compile(r'^[\d,]+(원|won|krw)?$')  # 숫자·콤마·원·won·krw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

cfg = configparser.ConfigParser()
cfg.read("omniparser/config.ini", encoding="utf-8")
VISION_GRID_N: int = int(cfg["User"].get("vision_grid", 8))

MOVE_OPTIONS: Tuple[Tuple[int, int], ...] = (
    (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, 0)
)
TOTAL_ACTIONS = len(MOVE_OPTIONS)

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ────────────────────────────────────────────────────────────────────────────────
# Reward & Curriculum Config
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """보상 항목 및 스케일 정의"""
    move_cost: float = -0.05
    dist_penalty: float = -0.03
    exploration_bonus: float = 0.02
    dup_penalty: float = -0.4
    semantic_reward: float = 0.2
    goal_reward: float = 3.0
    edge_penalty: float = -0.15
    edge_cap: int = 3
    coverage_coef: float = 0.05
    empty_penalty: float = -0.15
    density_coef: float = 0.15
    token_density_max: int = 10
    sal_coef: float = 0.12
    dir_coef: float = 0.02
    # KL
    kl_beta_max: float = 0.3  # curriculum 상한
    # Hint / Curriculum
    hint_coef: float = 0.3
    hint_steps: int = 10
    ag_bonus_coef: float = 0.03
    default_cover_end: float = 0.6

RC = RewardConfig()

@dataclass
class HP:
    lr: float = 2.5e-4
    batch_size: int = 512
    epochs_per_update: int = 6
    gamma: float = 0.99
    lam: float = 0.95
    total_episodes: int = 1_700

# ────────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ────────────────────────────────────────────────────────────────────────────────

def build_vocab_index(vocab: List[str]) -> Dict[str, int]:
    return {t: i for i, t in enumerate(vocab)}

def tokens_to_bow(tokens: List[str], vdx: Dict[str, int]):
    v = np.zeros(len(vdx), np.float32)
    for t in tokens:
        if t in vdx:
            v[vdx[t]] = 1.0
    return v

def bbox_center(bb):
    if isinstance(bb[0], (list, tuple)):
        x = sum(p[0] for p in bb) / 4; y = sum(p[1] for p in bb) / 4
    else:
        x = (bb[0] + bb[2]) / 2; y = (bb[1] + bb[3]) / 2
    return x, y

def is_inside(bb, box):
    x, y = bbox_center(bb); x1, y1, x2, y2 = box
    return x1 <= x < x2 and y1 <= y < y2


def find_goal_tokens(goal, texts, boxes, max_extra: int = 10):
    goal_n = normalize_token(goal)
    print(f"goal_n: {goal_n}")

    cands = []
    for t, bb in zip(texts, boxes):
        t_n = normalize_token(t)
        print(f"  token: {t}, norm: {t_n}")
        if goal_n in t_n:
            print(f"    goal_n in token!")
            exact = (t_n == goal_n)
            diff  = abs(len(t_n) - len(goal_n))
            tail  = t_n[len(goal_n):]

            prefix = False
            if (t_n.startswith(goal_n) and diff <= max_extra and
                (not tail.isalpha() or CURRENCY_TAIL_RE.match(tail))):
                prefix = True
                print(f"    prefix ok: {t_n}")

            if exact or prefix:
                print(f"    candidate: {t}")
                cands.append((exact, diff, t, bb))

    cands.sort(key=lambda x: (-x[0], x[1]))
    print(f"cands: {[(t, diff) for exact, diff, t, bb in cands]}")
    return [(t, bb) for _, _, t, bb in cands]

# ─── Saliency ──────────────────────────────────────────────────────────────

def compute_saliency_map(img: Image.Image, N: int = 8):
    """Spectral Residual + HSV backup"""
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    ok, smap = False, None
    try:
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, smap = sal.computeSaliency(img_bgr)
    except Exception:
        ok = False
    if not ok or smap is None:
        # backup : V‑채널 gradient magnitude
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
        smap = np.hypot(gx, gy)
    smap = cv2.resize(smap.astype(np.float32), (N, N), cv2.INTER_AREA)
    smap -= smap.min(); smap /= smap.max() + 1e-6
    return smap

# ─── Human Prior (gaussian‑like) ───────────────────────────────────────────

def default_human_prior(N: int = 8, sigma: float = 0.8):
    Y, X = np.mgrid[0:N, 0:N]; cx, cy = (N - 1) / 2, N * 0.3
    dist = np.hypot((X - cx) / (N / 2), (Y - cy) / (N / 2))
    prior = np.exp(-(dist ** 2) / sigma); prior /= prior.sum()
    return prior

# ────────────────────────────────────────────────────────────────────────────────
# Appraisal Module (Elderly V2 – unchanged except typing tweaks)
# ────────────────────────────────────────────────────────────────────────────────

class AppraisalModuleElderlyV2:
    HIGH_CONTRAST_COLORS = {"red", "yellow", "white", "빨강", "노랑"}
    FOVEAL_DEG = 10
    CRT_SLOPE_MS_PER_YEAR = 0.0028
    PROPRIO_ERROR_THRESH_CM = 3.0

    def __init__(self, env: "GazeKioskEnv"):
        self.env = env
        self.base_cert = (0.25, 0.55)

    def compute(
        self,
        grid_idx: Optional[Tuple[int, int]],
        in_view_tokens: List[str],
        visited_ratio: float,
        age_years: int,
        proprio_endpt_err_cm: float,
    ) -> np.ndarray:

        # 1. motivational relevance
        # 시야가 좁고(foveal zone 위주), 고정 시간‧사카드 진폭이 짧으므로 화면 중심부에서 시야각 ≤ 10° 안에 존재하거나 
        # 고대비·큰 글자일 때만 급격히 가중치 상승.
        # Characteristics of Gaze Behavior and Associated Factors ...
        mot_rel = 0.0
        if grid_idx:
            vx, vy, vw, vh = self.env._grid_to_viewport(*grid_idx)
            cx, cy = vx + vw / 2, vy + vh / 2
            cdist = math.hypot(cx - self.env.W / 2, cy - self.env.H / 2)
            fov_r = math.tan(math.radians(self.FOVEAL_DEG)) * self.env.H / 2
            mot_rel = 1.0 - min(1.0, cdist / max(fov_r, 1.0))
            if any(tok.lower() in self.HIGH_CONTRAST_COLORS for tok in in_view_tokens):
                mot_rel = min(1.0, mot_rel + 0.15)
        
        # 2. certainty
        # 기본 범위를 0.25–0.55(인지 저하 반영)로 낮춤. 
        # 이후 CRT 지연(≈ 2.8 ms/년)과 반응 지연 latency를 반비례적으로 매 step 감산 → 느릴수록 certainty 급감.
        # Age-related slowing of response selection ...
        cert_low, cert_hi = self.base_cert
        crt_penalty = max(0.0, (age_years - 60) * self.CRT_SLOPE_MS_PER_YEAR)
        certainty = np.clip(random.uniform(cert_low, cert_hi) - crt_penalty, 0, 1)
        
        # 3. novelty
        # 기존 요소에 피로 누적 적용, 빈번한 시선 전환은 피로누적
        # 탐색 의욕 감소
        novelty = max(0.0, 1.0 - visited_ratio) * 0.5
        
        # 4. goal congruence
        # 목표와 뷰포트 거리 비례 노이즈, 고대비면 가산
        goal_cong = 1.0 if mot_rel > 0.8 else 0.0

        # 5. coping potential
        # 과제수행 능력감, 고령자는 End-Point Error, Initial Direction Error, Response Latency 모두 유의하게 높음
        # Aging increases proprioceptive error for a broad range of movement ...
        coping = 0.3 + 0.4 * visited_ratio
        if proprio_endpt_err_cm > self.PROPRIO_ERROR_THRESH_CM:
            coping = max(0.0, coping - 0.2)

        # 6. anticipation
        # 결정 지연을 고려함, 신뢰가 낮을수록 다음 행동에 대한 기대가 커진다는 고령자 행동 반영
        # Age-related slowing of response selection ...
        anticipation = (1 - certainty) * (0.8 + novelty / 2)
        vec = np.array(
            [mot_rel, certainty, novelty, goal_cong, coping, anticipation],
            dtype=np.float32,
        )
        return np.clip(vec + np.random.normal(0, 0.02, 6), 0, 1)

# ────────────────────────────────────────────────────────────────────────────────
# Environment
# ────────────────────────────────────────────────────────────────────────────────

class GazeKioskEnv:
    def __init__(
        self,
        img_path: str,
        goal_seq: List[str],
        *,
        human_prior: Optional[np.ndarray] = None,
        verbose: bool = True,
        use_appraisal: bool = True,
        cognitive_level: str = "normal",
        age_years: int = 70,
        reward_cfg: RewardConfig = RC,
        menu: Optional[List[str]] = None, 
        
    ) -> None:
        self.image = Image.open(img_path).convert("RGB")
        self.W, self.H = self.image.size
        self.N = VISION_GRID_N
        self.cell_w, self.cell_h = self.W / self.N, self.H / self.N
        self.max_steps = 90
        self.cognitive_level = cognitive_level
        self.use_appraisal = use_appraisal
        self.age_years = age_years
        self.reward_cfg = reward_cfg
        self._last_candidates: list = []

        # OCR
        if verbose:
            print("[Env] OCR running …")
        self.ocr_txt, self.ocr_bb = check_ocr_box(
            self.image, display_img=False, output_bb_format="xyxy", use_paddleocr=True
        )
        if verbose:
            print(f"OCR found {len(self.ocr_txt)} text boxes")

        if menu is None:
            menu = list({t.strip() for t in self.ocr_txt if t.strip()})
        self.menu = menu   

        # Static maps
        self.token_cells = np.zeros((self.N, self.N), bool)
        for bb in self.ocr_bb:
            x, y = bbox_center(bb)
            gx, gy = min(int(x / self.cell_w), self.N - 1), min(int(y / self.cell_h), self.N - 1)
            self.token_cells[gy, gx] = True
        self.hint_map = self.token_cells.astype(np.float32)
        self.total_token_cells = self.token_cells.sum()
        self.saliency_map = compute_saliency_map(self.image, self.N)
        self.human_prior = human_prior if human_prior is not None else default_human_prior(self.N)

        if use_appraisal:
            self.app = AppraisalModuleElderlyV2(self)

        # curriculum KL schedule
        self.kl_beta: float = 0.0
        self.cover_end_ratio = self.reward_cfg.default_cover_end

        self.vdx = build_vocab_index(menu)
        self.set_goal_sequence(goal_seq)
        self.reset()

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------
    def set_goal_sequence(self, seq):
        self.goal_seq = list(seq)

    def set_kl_beta(self, beta: float):
        self.kl_beta = beta

    # ------------------------------------------------------------------
    # Grid/Viewport helpers
    # ------------------------------------------------------------------
    def _grid_to_viewport(self, gx: int, gy: int):
        return gx * self.cell_w, gy * self.cell_h, self.cell_w, self.cell_h

    def _vbox(self):
        bw, bh = self.cell_w, self.cell_h
        cx, cy = int(self.pos[0] * self.W), int(self.pos[1] * self.H)
        return max(0, cx - bw / 2), max(0, cy - bh / 2), min(self.W, cx + bw / 2), min(
            self.H, cy + bh / 2
        )

    # ------------------------------------------------------------------
    # Reset/Step
    # ------------------------------------------------------------------
    def reset(self):
        self.gx, self.gy = self.N // 2, 0
        self.prev_gy = 0
        self.goal_idx = 0
        self.steps = 0
        self.done = False
        self.visited = np.zeros((self.N, self.N), int)
        self.edge_streak = -1
        self.proprio_err_cm = 0.0  # stub
        self._update_pos()
        self.appraisal = np.zeros(6, np.float32)
        return self._obs()

    def _update_pos(self):
        self.pos = [(self.gx + 0.5) / self.N, (self.gy + 0.5) / self.N]

    def _coverage_ratio(self):
        visited_token = np.logical_and(self.visited > 0, self.token_cells).sum()
        return visited_token / max(1, self.total_token_cells)

    def step(self, a: int):
        """하나의 grid step"""
        # Apply action
        dx, dy = MOVE_OPTIONS[a]
        self.gx = int(np.clip(self.gx + dx, 0, self.N - 1))
        self.gy = int(np.clip(self.gy + dy, 0, self.N - 1))
        self._update_pos()
        self.steps += 1
        self.visited[self.gy, self.gx] += 1
        vcnt = self.visited[self.gy, self.gx]

        rc = self.reward_cfg  # local alias
        r = rc.move_cost  # 기본 이동 비용

        # Δ거리 패널티 (대각선 이동 > 직선)
        r += rc.dist_penalty * max(0.0, math.hypot(dx, dy) - 1)

        # Saliency 보너스 + 방향성(Y 증분)
        r += rc.sal_coef * self.saliency_map[self.gy, self.gx]
        dir_y = self.gy - self.prev_gy
        if dir_y > 0:
            r += rc.dir_coef * (dir_y / (self.N - 1))
        self.prev_gy = self.gy

        # Hint steps
        if self.steps <= rc.hint_steps and self.hint_map[self.gy, self.gx]:
            r += rc.hint_coef

        # OCR in‑view 분석
        pairs      = [(t, b) for t, b in zip(self.ocr_txt, self.ocr_bb) if is_inside(b, self._vbox())]
        in_view    = [t for t, _ in pairs]   # ← 기존과 동일
        in_view_bb = [b for _, b in pairs]   # ← 새로 만든 변수

        #디버깅
        goal_tok = self.goal_seq[self.goal_idx] if self.goal_idx < len(self.goal_seq) else None
        print(f"step: {self.step}, goal: {goal_tok}, in_view: {in_view}")
        print(f"norm_in_view: {[normalize_token(t) for t in in_view]}")

        token_cnt = len(in_view)
        tokens_n = [normalize_token(t) for t in in_view]
        r += rc.empty_penalty if token_cnt == 0 else rc.density_coef * min(token_cnt, rc.token_density_max) / rc.token_density_max

        # Exploration / Duplication
        if self.token_cells[self.gy, self.gx]:
            r += rc.exploration_bonus if vcnt == 1 else rc.dup_penalty / math.sqrt(vcnt)

        # Edge 패널티
        edge = self.gx in (0, self.N - 1) or self.gy in (0, self.N - 1)
        self.edge_streak = self.edge_streak + 1 if edge else 0
        if edge:
            r += rc.edge_penalty * min(self.edge_streak, rc.edge_cap)

        # Coverage shaping
        visited_ratio = self._coverage_ratio()
        dens_scalar = min(token_cnt, rc.token_density_max) / rc.token_density_max if token_cnt else 0
        r += rc.coverage_coef * (1 - visited_ratio) * dens_scalar

        # Goal check
        found = False
        if self.goal_idx < len(self.goal_seq):
            goal   = self.goal_seq[self.goal_idx]
            goal_n = normalize_token(goal)

            cand   = find_goal_tokens(goal, in_view, in_view_bb, max_extra=10)
            print(cand)
            if cand:
                top_n = normalize_token(cand[0][0])
                # ① exact ② prefix 둘 다 성공으로 인정
                if top_n == goal_n or top_n.startswith(goal_n):
                    found = True

            if found:
                r += rc.semantic_reward + rc.goal_reward
                self.goal_idx += 1

            self._last_candidates = cand  
        
        # Appraisal shaping
        if self.use_appraisal:
            self.appraisal = self.app.compute(
                (self.gx, self.gy), in_view, visited_ratio, self.age_years, self.proprio_err_cm
            )
            r += rc.ag_bonus_coef * (self.appraisal[0] + self.appraisal[4])
        else:
            self.appraisal.fill(0.0)

        # ── KL Regularization (step‑wise) ───────────────────────────────
        if self.kl_beta > 0:
            p_step = 1.0 / (visited_ratio + 1e-6)  # crude surrogate (visited_ratio 증가 → p 감소)
            kl_penalty = self.kl_beta * p_step * self.human_prior[self.gy, self.gx]
            r -= kl_penalty

        # ── Termination ────────────────────────────────────────────────
        done = (
            visited_ratio >= self.cover_end_ratio
            or self.goal_idx >= len(self.goal_seq)
            or self.steps >= self.max_steps
        )
        if done and self.steps >= self.max_steps:
            r -= 5.0
            self.done = True

    

        return self._obs(), float(r), done, {"found": found}

    # ------------------------------------------------------------------
    def _obs(self):
        in_view = [t for t, b in zip(self.ocr_txt, self.ocr_bb) if is_inside(b, self._vbox())]
        sal_val = self.saliency_map[self.gy, self.gx]
        bow = tokens_to_bow(in_view, self.vdx)
        visited_bin = (self.visited > 0).astype(np.float32).flatten()
        base = np.concatenate(
            [
                [self.goal_idx / max(1, len(self.goal_seq)), *self.pos, sal_val],
                bow,
                visited_bin,
            ]
        )
        if self.use_appraisal:
            base = np.concatenate([base, self.appraisal])
        return base.astype(np.float32)

# ────────────────────────────────────────────────────────────────────────────────
# Actor‑Critic Network
# ────────────────────────────────────────────────────────────────────────────────

class GazeActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hid: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU()
        )
        self.actor = nn.Linear(hid, TOTAL_ACTIONS)
        self.critic = nn.Linear(hid, 1)

    def forward(self, x):
        z = self.net(x)
        return self.actor(z), self.critic(z).squeeze(-1)

# ────────────────────────────────────────────────────────────────────────────────
# GAE helper
# ────────────────────────────────────────────────────────────────────────────────

def compute_gae(rew, val, gamma=0.99, lam=0.95):
    val = np.concatenate([val, [0.0]])
    adv, gae = [], 0.0
    for t in reversed(range(len(rew))):
        delta = rew[t] + gamma * val[t + 1] - val[t]
        gae = delta + gamma * lam * gae
        adv.insert(0, gae)
    return adv, (val[:-1] + adv).tolist()

# ────────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────────

def train():
    writer = SummaryWriter(log_dir="runs/gaze_elderly_v3")

    env = GazeKioskEnv(
        "screen5.png",
        goal_seq=["dummy"],
        verbose=False,
        age_years=70,
    )
    sampler = lambda: random.sample(env.menu, min(5, len(env.menu)))

    obs_dim = env.reset().shape[0]
    agent = GazeActorCritic(obs_dim).to(device)
    opt = optim.Adam(agent.parameters(), HP.lr)

    ep_ret, ep_len = [], []
    ckpt_dir = "omniparser/gaze_logs_v8"; os.makedirs(ckpt_dir, exist_ok=True)

    for ep in range(HP.total_episodes):
        # curriculum KL 스케줄
        frac = min(1.0, ep / 1000)
        env.set_kl_beta(RC.kl_beta_max * frac)
        goals = sampler()              # ← 1) 목표를 변수에 저장
        env.set_goal_sequence(goals)   # ← 2) 환경에 전달
        obs = env.reset()

        if (ep + 1) % 20 == 0:
           print(f"[DBG] Epoch {ep+1:04d}  goals → {goals}")

        buf: List[Tuple[np.ndarray, int, float, float]] = []
        ret_ep = 0.0
        while True:
            logits, _ = agent(
                torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            )
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample().item()
            logp = dist.log_prob(torch.tensor(act, device=device)).item()
            nxt, r, done, _ = env.step(act)
            buf.append((obs, act, r, logp))
            obs = nxt; ret_ep += r
            if done:
                break

        obs_a, act_a, rew_a, logp_a = map(list, zip(*buf))
        with torch.no_grad():
            val_a = agent(torch.tensor(obs_a, dtype=torch.float32, device=device))[1].cpu().numpy()
        adv, ret = compute_gae(rew_a, val_a, HP.gamma, HP.lam)
        adv = (np.array(adv) - np.mean(adv)) / (np.std(adv) + 1e-8)

        # Tensor → GPU
        obs_t = torch.tensor(np.stack(obs_a), dtype=torch.float32, device=device)
        act_t = torch.tensor(act_a, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)
        logp_old = torch.tensor(logp_a, dtype=torch.float32, device=device)

        # PPO update
        for _ in range(HP.epochs_per_update):
            idx = torch.randperm(len(obs_t))
            for i in range(0, len(idx), HP.batch_size):
                mb = idx[i : i + HP.batch_size]
                logits, val = agent(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_t[mb])
                ratio = torch.exp(logp - logp_old[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_t[mb]
                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = (val - ret_t[mb]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pol_loss + 0.5 * val_loss - 0.01 * ent

                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()

        # logging
        ep_ret.append(ret_ep); ep_len.append(len(buf))
        writer.add_scalar("Return", ret_ep, ep)
        writer.add_scalar("EpisodeLen", len(buf), ep)
        if (ep + 1) % 50 == 0:
            avg_r = np.mean(ep_ret[-100:]); avg_l = np.mean(ep_len[-100:])
            print(f"[EP {ep+1}] avgR={avg_r:.2f}  avgLen={avg_l:.1f}")
            writer.add_scalar("avg_return", avg_r, ep)
            writer.add_scalar("avg_len", avg_l, ep)


    # ── 그래프 저장 (PNG) ───────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(ep_ret); plt.title("Return")
    plt.subplot(1, 2, 2); plt.plot(ep_len); plt.title("EpisodeLen")
    plt.tight_layout()
    png_path = os.path.join(ckpt_dir, "learning_curve.png")
    plt.savefig(png_path)
    print(f"Saved plot {png_path}")

    # final save
    torch.save(agent.state_dict(), "omniparser/gaze_ppo_v8.pt")
    writer.close()
    print("Training complete. Model & logs saved.")

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed(SEED)
    start = time.time()
    train()
    print(f"Total training time: {time.time() - start:.1f}s")
