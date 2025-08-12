# run_omniparser_with_ppo_v9.py
from __future__ import annotations

import os, random, math, time, configparser
from dataclasses import dataclass
# [CHANGED] AppraisalModuleElderlyV2가 Optional을 사용하므로 Optional 추가
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from utils.utils import check_ocr_box           # 사용자 util
from text_normalizer import normalize_token     # 선택적 전처리

# ───────────────────────────────────────────────────────────────
# [APPRAISAL-ADD] Elderly Appraisal 모듈 정의
# ───────────────────────────────────────────────────────────────
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
        cert_low, cert_hi = self.base_cert
        crt_penalty = max(0.0, (age_years - 60) * self.CRT_SLOPE_MS_PER_YEAR)
        certainty = np.clip(random.uniform(cert_low, cert_hi) - crt_penalty, 0, 1)
        
        # 3. novelty
        novelty = max(0.0, 1.0 - visited_ratio) * 0.5
        
        # 4. goal congruence
        goal_cong = 1.0 if mot_rel > 0.8 else 0.0

        # 5. coping potential
        coping = 0.3 + 0.4 * visited_ratio
        if proprio_endpt_err_cm > self.PROPRIO_ERROR_THRESH_CM:
            coping = max(0.0, coping - 0.2)

        # 6. anticipation
        anticipation = (1 - certainty) * (0.8 + novelty / 2)
        vec = np.array(
            [mot_rel, certainty, novelty, goal_cong, coping, anticipation],
            dtype=np.float32,
        )
        return np.clip(vec + np.random.normal(0, 0.02, 6), 0, 1)

# ───────────────────────────────────────────────────────────────
# Determinism & device
# ───────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ───────────────────────────────────────────────────────────────
# Config / Hyper-parameters
# ───────────────────────────────────────────────────────────────
cfg = configparser.ConfigParser()
cfg.read("omniparser/config.ini", encoding="utf-8")
VISION_GRID_N: int = int(cfg["User"].get("vision_grid", 9))  # 기본 9×9 = 81

MOVE_OPTIONS: Tuple[Tuple[int,int], ...] = (
    (0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)
)
TOTAL_ACTIONS = len(MOVE_OPTIONS)

@dataclass
class RewardConfig:
    # 셀 방문
    first_cell      : float = +0.25   # 빈칸이라도 첫 방문
    revisit_cell    : float = -0.10

    # 텍스트 특별 가중
    first_token_bonus: float = +0.20
    dist_shaping     : float = +0.30  # 0.30/(1+d)

    # 연속 커버리지 shaping
    step_cov_bonus  : float = +0.10   # 새 셀마다

    # 에피소드
    max_steps       : int   = 2000
    coverage_bonus  : float = +2.0    # 모든 셀 방문 성공

    hint_weight: float = 0.3
RC = RewardConfig()

@dataclass
class HP:
    lr               : float = 2.5e-4
    batch_size       : int = 512
    epochs_per_update: int = 4
    gamma            : float = 0.99
    lam              : float = 0.95
    total_episodes   : int = 2500
    ent_coef         : float = 0.08

# ───────────────────────────────────────────────────────────────
# Utility helpers
# ───────────────────────────────────────────────────────────────
def build_vocab_index(words: List[str]) -> Dict[str,int]:
    return {w:i for i,w in enumerate(words)}

def tokens_to_bow(tokens: List[str], vdx: Dict[str,int]):
    v = np.zeros(len(vdx), np.float32)
    for t in tokens:
        if t in vdx: v[vdx[t]] = 1.0
    return v

def bbox_center(bb):
    if isinstance(bb[0], (list,tuple)):
        x = sum(p[0] for p in bb)/4; y = sum(p[1] for p in bb)/4
    else:
        x = (bb[0]+bb[2])/2; y = (bb[1]+bb[3])/2
    return x,y

def is_inside(bb, box):
    x,y = bbox_center(bb); x1,y1,x2,y2 = box
    return x1<=x<x2 and y1<=y<y2

def compute_saliency_map(img:Image.Image, N:int=9):
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    ok,smap = False,None
    try:
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok,smap = sal.computeSaliency(img_bgr)
    except Exception: ok=False
    if not ok or smap is None:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2].astype(np.float32)/255.
        gx = cv2.Sobel(v,cv2.CV_32F,1,0,ksize=3)
        gy = cv2.Sobel(v,cv2.CV_32F,0,1,ksize=3)
        smap = np.hypot(gx,gy)
    smap = cv2.resize(smap.astype(np.float32),(N,N),cv2.INTER_AREA)
    smap -= smap.min(); smap /= smap.max()+1e-6
    return smap

# ───────────────────────────────────────────────────────────────
# Environment
# ───────────────────────────────────────────────────────────────
class GazeKioskEnv:
    def __init__(self, img_path:str, *, reward_cfg:RewardConfig=RC, verbose=True):
        self.image = Image.open(img_path).convert("RGB")
        self.W,self.H = self.image.size
        self.N = VISION_GRID_N
        self.cell_w, self.cell_h = self.W/self.N, self.H/self.N
        self.reward_cfg = reward_cfg
        self.max_steps = reward_cfg.max_steps

        self.saliency_map = np.zeros((self.N, self.N), dtype=np.float32)
        self.hint_map     = np.zeros((self.N, self.N), dtype=np.float32)
        self.prev_gx, self.prev_gy = None, None  # [추가] 이전 위치 저장
        self.prev_dir = None  # [추가] 이전 이동 방향
        self.line_visited = set()  # [추가] 라인 커버리지 보상용
        self.col_visited = set()   # [추가] 열 커버리지 보상용
        self.stuck_counter = 0  # [추가] 한 곳에 머무는 시간 카운터
        self.last_coverage_reset = 0  # [추가] 마지막 커버리지 리셋 스텝

        # OCR
        if verbose: print("[Env] OCR…")
        self.ocr_txt, self.ocr_bb = check_ocr_box(
            self.image, display_img=False, output_bb_format="xyxy", use_paddleocr=True
        )
        if verbose: print(f"[Env] {len(self.ocr_txt)} boxes")

        # 텍스트 셀 마스크
        self.token_cells = np.zeros((self.N,self.N), bool)
        for bb in self.ocr_bb:
            x,y = bbox_center(bb)
            gx = min(int(x/self.cell_w), self.N-1)
            gy = min(int(y/self.cell_h), self.N-1)
            self.token_cells[gy,gx] = True
        self.total_cells = self.N*self.N

        self.saliency_map = compute_saliency_map(self.image,self.N)
        vocab = list({t.strip() for t in self.ocr_txt if t.strip()})
        self.vdx = build_vocab_index(vocab)

        # [APPRAISAL-ADD] ---- Elderly Appraisal 모듈 및 파라미터 ----
        self.appraisal_mod = AppraisalModuleElderlyV2(self)
        self.appraisal_age = int(os.environ.get("APPRAISAL_AGE", "70"))               # 기본 70세
        self.proprio_endpt_err_cm = float(os.environ.get("APPRAISAL_PROPRIO_ERR_CM", "4.0"))
        self.appraisal_mode = os.environ.get("APPRAISAL_MODE", "RSv1")                 # "off" | "RSv1"
        # [APPRAISAL-ADD] -----------------------------------------------

        self.reset()

    # helpers
    def _vbox(self):
        bx,by = int(self.pos[0]*self.W), int(self.pos[1]*self.H)
        w,h = self.cell_w,self.cell_h
        return max(0,bx-w/2), max(0,by-h/2), min(self.W,bx+w/2), min(self.H,by+h/2)

    # [APPRAISAL-ADD] grid 인덱스를 뷰포트(px) 박스로 변환
    def _grid_to_viewport(self, gx:int, gy:int):
        x1 = gx * self.cell_w
        y1 = gy * self.cell_h
        x2 = (gx + 1) * self.cell_w
        y2 = (gy + 1) * self.cell_h
        return x1, y1, x2, y2

    # [APPRAISAL-ADD] 현재 뷰포트 안의 토큰 텍스트
    def _in_view_tokens(self) -> List[str]:
        return [t for t,b in zip(self.ocr_txt, self.ocr_bb) if is_inside(b, self._vbox())]

    def _grid_cov(self):                 # 전체 커버리지
        return (self.visited>0).sum()/self.total_cells

    # reset / step
    def reset(self):
        self.gx,self.gy = np.random.randint(0,self.N,size=2)
        self.pos = [(self.gx+0.5)/self.N, (self.gy+0.5)/self.N]
        self.visited = np.zeros((self.N,self.N), int)
        self.steps = 0; self.prev_cov = 0.0
        self.prev_gx, self.prev_gy = None, None  # [추가]
        self.prev_dir = None  # [추가]
        self.line_visited = set()  # [추가]
        self.col_visited = set()   # [추가]
        self.stuck_counter = 0  # [추가]
        self.last_coverage_reset = 0  # [추가]
        
        return self._obs()

    def reset_scan(self):
        """시야 스캔 전용 상태만 초기화(visited·steps 등)"""
        self.visited.fill(0)
        self.steps     = 0
        self.prev_cov  = 0.0
        return self._obs()

    def step(self, a:int):
        rc = self.reward_cfg
        # [추가] 이전 위치/방향 저장
        prev_gx, prev_gy = self.gx, self.gy
        prev_dir = self.prev_dir
        # move
        dx,dy = MOVE_OPTIONS[a]
        self.gx = int(np.clip(self.gx+dx,0,self.N-1))
        self.gy = int(np.clip(self.gy+dy,0,self.N-1))
        self.pos = [(self.gx+0.5)/self.N, (self.gy+0.5)/self.N]
        self.steps += 1
        # [추가] 이동 거리 계산
        move_dist = abs(self.gx - prev_gx) + abs(self.gy - prev_gy) if prev_gx is not None else 0
        # [추가] 이동 방향(정수 튜플)
        move_dir = (np.sign(self.gx - prev_gx) if prev_gx is not None else 0, np.sign(self.gy - prev_gy) if prev_gy is not None else 0)
        self.prev_gx, self.prev_gy = self.gx, self.gy
        self.prev_dir = move_dir
        self.visited[self.gy,self.gx] += 1
        vcnt = self.visited[self.gy,self.gx]
        is_token = bool(self.token_cells[self.gy,self.gx])
        r = rc.first_cell if vcnt==1 else rc.revisit_cell
        if is_token and vcnt==1:
            r += rc.first_token_bonus
        r += rc.hint_weight * float(self.hint_map[self.gy, self.gx])  # 새 항목
        # 거리 shaping (가장 가까운 미방문 텍스트)
        unvis_tok = self.token_cells & (self.visited==0)
        if unvis_tok.any():
            tgt_gy, tgt_gx = np.column_stack(np.where(unvis_tok))[0]
            dist = abs(tgt_gx-self.gx)+abs(tgt_gy-self.gy)
            r += rc.dist_shaping/(1+dist)
        # saliency
        r += 0.1*self.saliency_map[self.gy,self.gx]
        # 새 셀 커버리지 shaping
        new_cov = self._grid_cov()
        if new_cov>self.prev_cov:
            r += rc.step_cov_bonus
        self.prev_cov = new_cov
        # [APPRAISAL-ADD] Elderly Appraisal 계산 및 선택적 보상 셰이핑
        in_view_tokens = self._in_view_tokens()
        app_vec = self.appraisal_mod.compute(
            (self.gx, self.gy),
            in_view_tokens,
            new_cov,
            self.appraisal_age,
            self.proprio_endpt_err_cm,
        )
        if self.appraisal_mode == "RSv1":
            r -= 0.01 * (1.0 - float(app_vec[0]))
        # [추가1] 라인(행/열) 전체 방문 시 보상
        if (self.gy not in self.line_visited) and (self.visited[self.gy,:]>0).all():
            r += 0.3  # 라인 커버리지 보상(값 완화)
            self.line_visited.add(self.gy)
        if (self.gx not in self.col_visited) and (self.visited[:,self.gx]>0).all():
            r += 0.3  # 열 커버리지 보상(값 완화)
            self.col_visited.add(self.gx)
        # [추가2] 이동 거리 패널티(멀리 점프할수록 패널티)
        if move_dist > 1:
            r -= 0.05 * (move_dist-1)  # 1칸 초과 이동마다 -0.05로 완화
        # [추가3] 중복 방문 패널티 강화 (커버리지 리셋 후에는 완화)
        if vcnt > 1:
            # 커버리지 리셋 후 100 스텝 동안은 패널티 완화
            if self.steps - self.last_coverage_reset < 100:
                r -= 0.01 * (vcnt-1)  # 패널티 완화
            else:
                r -= 0.02 * (vcnt-1)  # 기존 패널티
        # [추가4] 이동 방향 연속성 보상(이전 이동 방향과 같으면 보상)
        if prev_dir is not None and move_dir == prev_dir and move_dir != (0,0):
            r += 0.02  # 같은 방향 연속 이동 보상(값 완화)
        
        # [추가5] 커버리지 100% 도달 시 초기화 및 재탐색
        done = False
        if new_cov >= 1.0:                 # 전 셀 방문
            r += rc.coverage_bonus
            # 커버리지 초기화하여 재탐색 유도
            self.visited.fill(0)
            self.prev_cov = 0.0
            self.line_visited.clear()
            self.col_visited.clear()
            self.stuck_counter = 0
            self.last_coverage_reset = self.steps
            # 새로운 시작점으로 랜덤 이동
            self.gx, self.gy = np.random.randint(0, self.N, size=2)
            self.pos = [(self.gx+0.5)/self.N, (self.gy+0.5)/self.N]
            self.prev_gx, self.prev_gy = None, None
            self.prev_dir = None
            print(f"[COVERAGE RESET] Step {self.steps}: Coverage reset, new position ({self.gx}, {self.gy})")
        
        elif self.steps >= rc.max_steps:
            r -= 1.0; done = True
        return self._obs(), float(r), done, {
            "grid_cov": new_cov,
            "steps": self.steps,
            "appraisal": app_vec,   # [APPRAISAL-ADD] info에 벡터 포함
        }

    # ── 관측 ──────────────────────────────────────────
    def _obs(self):
        # [CHANGED] in_view 토큰 수집을 헬퍼로 통일
        in_view = self._in_view_tokens()
        bow = tokens_to_bow(in_view,self.vdx)
        visited_bin = (self.visited>0).astype(np.float32).flatten()

        unvis_tok = self.token_cells & (self.visited==0)
        if unvis_tok.any():
            tgt_gy, tgt_gx = np.column_stack(np.where(unvis_tok))[0]
            dir_vec = np.array([tgt_gx-self.gx, tgt_gy-self.gy], np.float32)/self.N
        else:
            dir_vec = np.zeros(2,np.float32)

        hint_flat = self.hint_map.astype(np.float32).flatten()  # 추가

        # [APPRAISAL-ADD] Elderly Appraisal 6D (mot_rel, certainty, novelty, goal_cong, coping, anticipation)
        app_vec = self.appraisal_mod.compute(
            (self.gx, self.gy),
            in_view,
            self._grid_cov(),
            self.appraisal_age,
            self.proprio_endpt_err_cm,
        ).astype(np.float32)

        vec = np.concatenate([
            self.pos,                   # 2
            [self._grid_cov()],         # 1
            dir_vec,                    # 2
            bow,                        # |V|
            visited_bin,                # N*N
            hint_flat,                  # N*N
            app_vec                     # [APPRAISAL-ADD] 6
        ])
        return vec.astype(np.float32)

# ───────────────────────────────────────────────────────────────
# Actor-Critic
# ───────────────────────────────────────────────────────────────
class GazeActorCritic(nn.Module):
    def __init__(self, obs_dim:int, hid:int=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,hid), nn.ReLU(),
            nn.Linear(hid,hid), nn.ReLU()
        )
        self.actor = nn.Linear(hid,TOTAL_ACTIONS)
        self.critic= nn.Linear(hid,1)
    def forward(self,x):
        z = self.net(x)
        return self.actor(z), self.critic(z).squeeze(-1)

# ───────────────────────────────────────────────────────────────
# GAE
# ───────────────────────────────────────────────────────────────
def compute_gae(rew,val,gamma,lam):
    val = np.concatenate([val,[0.0]])
    adv,gae=[],0.0
    for t in reversed(range(len(rew))):
        delta = rew[t]+gamma*val[t+1]-val[t]
        gae = delta + gamma*lam*gae
        adv.insert(0,gae)
    return adv,(val[:-1]+adv).tolist()

# ───────────────────────────────────────────────────────────────
# Training loop
# ───────────────────────────────────────────────────────────────
def train():
    writer = SummaryWriter("runs/gaze_scan_v2")
    env = GazeKioskEnv(img_path="screen5.png", verbose=False)
    obs_dim = env.reset().shape[0]
    agent = GazeActorCritic(obs_dim).to(device)
    opt = optim.Adam(agent.parameters(), HP.lr)

    ep_ret,ep_len = [],[]
    os.makedirs("omniparser/gaze_logs_v9",exist_ok=True)

    for ep in range(HP.total_episodes):
        obs = env.reset(); buf=[]; ret_ep=0.0
        while True:
            logits,_ = agent(torch.tensor(obs,dtype=torch.float32,device=device).unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample().item()
            logp = dist.log_prob(torch.tensor(act,device=device)).item()
            nxt,r,done,info = env.step(act)
            buf.append((obs,act,r,logp))
            obs = nxt; ret_ep += r
            if done: break

        obs_a,act_a,rew_a,logp_a = map(list,zip(*buf))
        with torch.no_grad():
            val_a = agent(torch.tensor(obs_a,dtype=torch.float32,device=device))[1].cpu().numpy()
        adv,ret = compute_gae(rew_a,val_a,HP.gamma,HP.lam)
        adv = (np.array(adv)-np.mean(adv))/(np.std(adv)+1e-8)

        obs_t = torch.tensor(np.stack(obs_a),dtype=torch.float32,device=device)
        act_t = torch.tensor(act_a,device=device)
        adv_t = torch.tensor(adv,dtype=torch.float32,device=device)
        ret_t = torch.tensor(ret,dtype=torch.float32,device=device)
        logp_o= torch.tensor(logp_a,dtype=torch.float32,device=device)

        for _ in range(HP.epochs_per_update):
            idx = torch.randperm(len(obs_t))
            for i in range(0,len(idx),HP.batch_size):
                mb = idx[i:i+HP.batch_size]
                logits,val = agent(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_t[mb])
                ratio = torch.exp(logp-logp_o[mb])
                surr1 = ratio*adv_t[mb]
                surr2 = torch.clamp(ratio,0.8,1.2)*adv_t[mb]
                pol_loss = -torch.min(surr1,surr2).mean()
                val_loss = (val-ret_t[mb]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pol_loss + 0.5*val_loss - HP.ent_coef*ent
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(),0.5); opt.step()

        ep_ret.append(ret_ep); ep_len.append(len(buf))
        writer.add_scalar("Return",ret_ep,ep)
        writer.add_scalar("EpisodeLen",len(buf),ep)
        if (ep+1)%50==0:
            print(f"[EP {ep+1}] avgR={np.mean(ep_ret[-50:]):.2f} avgLen={np.mean(ep_len[-50:]):.1f}")

    torch.save(agent.state_dict(),"omniparser/gaze_ppo_v9.pt")
    writer.close()
    print("Training done.")

if __name__=="__main__":
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)
    t0=time.time(); train(); print(f"Runtime: {time.time()-t0:.1f}s")
