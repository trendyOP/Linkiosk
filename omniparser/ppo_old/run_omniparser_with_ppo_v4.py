# gaze_ppo_refactor_v4.py  (TRAIN, 완전 통합본)
# -------------------------------------------------------
"""
● VISION_GRID_N = 6
● obs_dim = 3(goal 비율 + pos) + 6(bow) + 36(visited) = 45
● 연속 Edge 패널티 / Coverage 보너스
● dtype 오류·메서드 충돌 수정
● 리워드 그래프 → omniparser/gaze_logs_v4/learning_curve.png
"""
import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Dict
from PIL import Image
from utils.utils import check_ocr_box

# ───── 하이퍼파라미터 ────────────────────────────────────
VISION_GRID_N, TOTAL_ACTIONS = 8, 4
STEP_SIZE                     = 1 / VISION_GRID_N
MAX_STEPS                     = 50
MOVE_COST                     = -0.005
EXPLORATION_BONUS             = 0.02
DUP_PENALTY                   = -0.05
SEMANTIC_REWARD               = 0.2
GOAL_REWARD                   = 1.5
EDGE_PENALTY                  = -0.10
EDGE_CAP                      = 5        # NEW – cap after 5 streaks
COVERAGE_COEF                 = 0.05
COVER_END_THRESH              = 0.9
# new
EMPTY_PENALTY                 = -0.10
DENSITY_COEF                  = 0.05
TOKEN_DENSITY_MAX             = 10         # 토큰 10개 이상이면 만 보상
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ───── BoW util ─────────────────────────────────────────
def build_vocab_index(vocab: List[str]) -> Dict[str,int]:
    return {t:i for i,t in enumerate(vocab)}
def tokens_to_bow(tokens, vdx):
    v = np.zeros(len(vdx), np.float32)
    for t in tokens:
        idx = vdx.get(t)
        if idx is not None: v[idx] = 1.0
    return v

def bbox_center(bb):
    if isinstance(bb[0],(list,tuple)): 
        x=sum(p[0] for p in bb)/4
        y=sum(p[1] for p in bb)/4

    else: 
        x=(bb[0]+bb[2])/2
        y=(bb[1]+bb[3])/2
    return x,y

def is_inside(bb,box): 
    x,y=bbox_center(bb); 
    x1,y1,x2,y2=box; 
    return x1<=x<x2 and y1<=y<y2

# ───── 환경 ─────────────────────────────────────────────
class GazeKioskEnv:
    def __init__(self, img_path, menu, goal_seq):
        self.image = Image.open(img_path)
        self.N = VISION_GRID_N
        self.step_size = 1 / self.N                 # ★ 이름 충돌 해결
        self.max_steps = MAX_STEPS
        self.menu = menu
        self.vdx  = build_vocab_index(menu)

        print("[Env] OCR 수행 중…")
        self.ocr_txt, self.ocr_bb = check_ocr_box(
            self.image, display_img=False, output_bb_format='xyxy', use_paddleocr=True
        )

        # pre‑compute which grid cells contain any OCR token
        self.token_cells = np.zeros((self.N, self.N), bool)
        for bb in self.ocr_bb:
            x, y = bbox_center(bb)
            gx = min(int(x / (1920 / self.N)), self.N - 1)
            gy = min(int(y / (1080 / self.N)), self.N - 1)
            self.token_cells[gy, gx] = True
        self.total_token_cells = self.token_cells.sum()

        self.set_goal_sequence(goal_seq)
        self.reset()

    def set_goal_sequence(self, seq): 
        self.goal_seq = list(seq)

    def reset(self):
        self.goal_idx = self.steps = 0
        self.done = False
        self.pos = [0.5, 0.5]
        self.visited = np.zeros((self.N, self.N), int)
        self.edge_streak = 0
        return self._obs()

    def step(self, a: int):
        # move
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][a]
        dx *= self.step_size;  dy *= self.step_size
        self.pos[0] = np.clip(self.pos[0] + dx, 0, 1)
        self.pos[1] = np.clip(self.pos[1] + dy, 0, 1)
        self.steps += 1

        gy = min(int(self.pos[1] * self.N), self.N - 1)
        gx = min(int(self.pos[0] * self.N), self.N - 1)
        self.visited[gy, gx] += 1
        vcnt = self.visited[gy, gx]

        # ---------------- Reward ----------------
        # ----- observation helpers -----
        in_view = [t for t, b in zip(self.ocr_txt, self.ocr_bb) if is_inside(b, self._vbox())]
        token_cnt = len(in_view)
        density_scalar = min(token_cnt, TOKEN_DENSITY_MAX) / TOKEN_DENSITY_MAX

        # (0) base move cost
        r = MOVE_COST

        # (1) empty‑penalty & density‑reward
        if token_cnt == 0:
            r += EMPTY_PENALTY
        else:
            density = min(token_cnt, TOKEN_DENSITY_MAX) / TOKEN_DENSITY_MAX
            r += DENSITY_COEF * density

        # (2) exploration / duplicate only if token exists in that cell
        if self.token_cells[gy, gx]:
            if vcnt == 1:
                r += EXPLORATION_BONUS
            else:
                r += DUP_PENALTY / np.sqrt(vcnt)

        # (3) edge penalty with cap
        edge = gx in (0, self.N - 1) or gy in (0, self.N - 1)
        self.edge_streak = self.edge_streak + 1 if edge else 0
        if edge:
            r += EDGE_PENALTY * min(self.edge_streak, EDGE_CAP)

        # (4) coverage bonus weighted by density
        token_cells_visited = (self.visited & self.token_cells).astype(bool).sum()
        cov = 1 - token_cells_visited / max(1, self.total_token_cells)
        r += COVERAGE_COEF * cov * density_scalar

        # (5) goal reward (같음)
        goal = self.goal_seq[self.goal_idx]
        found = False
        if any(goal in t for t in in_view):
            r += SEMANTIC_REWARD + GOAL_REWARD
            self.goal_idx += 1
            self.visited[:] = 0
            found = True

        # (6) done 조건
        if token_cells_visited >= self.total_token_cells * COVER_END_THRESH \
           or self.goal_idx >= len(self.goal_seq) \
           or self.steps >= self.max_steps:
            self.done = True

        return self._obs(), r, self.done, {"found": found}

    # ---- helpers
    def _vbox(self):
        W,H = 1920,1080; bw,bh = W/self.N, H/self.N
        cx,cy = int(self.pos[0]*W), int(self.pos[1]*H)
        return max(0,cx-bw/2), max(0,cy-bh/2), min(W,cx+bw/2), min(H,cy+bh/2)

    def _obs(self):
        in_view=[t for t,b in zip(self.ocr_txt,self.ocr_bb) if is_inside(b,self._vbox())]
        token_cnt = len(in_view)
        density_scalar = min(token_cnt, TOKEN_DENSITY_MAX) / TOKEN_DENSITY_MAX
        bow = tokens_to_bow(in_view,self.vdx)
        return np.concatenate([
            [self.goal_idx / len(self.goal_seq), *self.pos, density_scalar],
            bow,
            self.visited.flatten()
        ]).astype(np.float32)

# ───── 모델 ─────────────────────────────────────────────
class GazeActorCritic(nn.Module):
    def __init__(self, obs_dim, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim,hid), nn.ReLU(),
                                 nn.Linear(hid,hid), nn.ReLU())
        self.actor  = nn.Linear(hid, TOTAL_ACTIONS)
        self.critic = nn.Linear(hid, 1)
    def forward(self, x):
        z = self.net(x)
        return self.actor(z), self.critic(z).squeeze(-1)

# ───── GAE ─────────────────────────────────────────────
def compute_gae(r, v, g=.99, lmb=.95):
    v = v + [0]; adv,gae=[],0
    for t in reversed(range(len(r))):
        delta = r[t] + g * v[t+1] - v[t]
        gae   = delta + g * lmb * gae
        adv.insert(0, gae)
    return adv, [a+v for a,v in zip(adv, v[:-1])]

# ───── 학습 루프 ───────────────────────────────────────
def train():
    menu = ['커피','주문','아이스크림','진정한고구마','진정한티라미수','메뉴']
    sampler = lambda: random.sample(menu, 3)
    env = GazeKioskEnv("screen2.png", menu, sampler())
    obs_dim = env.reset().shape[0]

    agent = GazeActorCritic(obs_dim).to(device)
    opt   = optim.Adam(agent.parameters(), 2.5e-4)

    ep_r_history, ep_len_history = [], []

    TOTAL_EP = 3500
    for ep in range(TOTAL_EP):
        env.set_goal_sequence(sampler())
        obs = env.reset()
        buf = []; ep_return=0

        done=False
        while not done:
            logits,_ = agent(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            dist = torch.distributions.Categorical(logits=logits)
            act  = dist.sample().item()
            logp = dist.log_prob(torch.tensor(act, device=device)).item()

            nxt,r,done,_ = env.step(act)
            buf.append((obs,act,r,logp))
            obs = nxt
            ep_return += r

        # --- GAE
        obs_arr, act_arr, rew_arr, logp_arr = list(zip(*buf))
        val_arr = [agent(torch.tensor(o,dtype=torch.float32,device=device).unsqueeze(0))[1].item() for o in obs_arr]
        adv, ret = compute_gae(rew_arr, val_arr)

        obs_t = torch.tensor(np.array(obs_arr), dtype=torch.float32, device=device)
        acts  = torch.tensor(act_arr, device=device)
        adv   = torch.tensor(adv, dtype=torch.float32, device=device)
        ret   = torch.tensor(ret, dtype=torch.float32, device=device)
        logp_old = torch.tensor(logp_arr, dtype=torch.float32, device=device)

        for _ in range(6):                          # PPO epochs
            idx = torch.randperm(len(obs_t))
            for i in range(0, len(idx), 32):        # mini-batch
                mb = idx[i:i+32]
                logits, val = agent(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(acts[mb])
                ratio= torch.exp(logp - logp_old[mb])
                s1   = ratio * adv[mb]
                s2   = torch.clamp(ratio,0.8,1.2) * adv[mb]
                policy_loss = -torch.min(s1,s2).mean()
                value_loss  = (val - ret[mb]).pow(2).mean()
                entropy     = dist.entropy().mean()
                loss = policy_loss + 0.5*value_loss - 0.01*entropy
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(),0.5); opt.step()

        ep_r_history.append(ep_return)
        ep_len_history.append(len(buf))
        if (ep+1) % 100 == 0:
            print(f"[EP {ep+1}] avgR={np.mean(ep_r_history[-100:]):.2f}  avgLen={np.mean(ep_len_history[-100:]):.1f}")

    # --- 그래프 저장
    os.makedirs("omniparser/gaze_logs_v4", exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(ep_r_history);   plt.title("Episode Reward")
    plt.subplot(1,2,2); plt.plot(ep_len_history); plt.title("Episode Length")
    plt.tight_layout(); plt.savefig("omniparser/gaze_logs_v4/learning_curve.png")

    torch.save(agent.state_dict(), "omniparser/gaze_ppo_v4.pt")
    print("saved gaze_ppo_v4.pt  +  learning_curve.png")

if __name__ == "__main__":
    train()
