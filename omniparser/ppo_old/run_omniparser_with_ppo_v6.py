import os, random, math
import numpy as np, cv2, torch
import torch.nn as nn, torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict        
from PIL import Image
import configparser     

from utils.utils import check_ocr_box

# INI 파일 읽기
config = configparser.ConfigParser()
config.read('omniparser/config.ini', encoding='utf-8')

# ───── 하이퍼파라미터 ──────────────────────────────────────────────────────
VISION_GRID_N   = int(config['User']['vision_grid'])
MOVE_OPTIONS    = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,0)]
TOTAL_ACTIONS   = len(MOVE_OPTIONS)        # 9 

MAX_STEPS       = 90        # 하나의 과업 시나리오 당 최대 움직임 횟수
MOVE_COST       = -0.05     # 움직일때 마다 보상 감소
DIST_PENALTY    = -0.03     # 목표와 멀어질수록 더 큰 보상 감소      
EXPLORATION_BONUS = 0.02    # 단, 처음 가보는 그리드에 대해선 탐험 보상을 줘 너무 한곳에 집중되지 않게 함
DUP_PENALTY     = -0.4      # 중복되는 셀을 자주 방문할수록, 보상 감소
SEMANTIC_REWARD = 0.2       # 추가 보상
GOAL_REWARD     = 3.0       # 목표 타겟을 찾을 경우 보상
EDGE_PENALTY    = -0.15     # 화면 가장자리에 있을수록 큰 감점, 단 현재 키오스크(좌 우에 빈칸이 많음)에만 적용
EDGE_CAP        = 3         # 가장자리에 오래 있을수록, 최대 3배의 edge_penalty 부여
COVERAGE_COEF   = 0.05      # 값이 커지면 탐색 유인 증가, 작아지면 보수적인 탐색 유도
COVER_END_THRESH= 4 #0.7      # 값이 커지면 더 많은 셀을 찾아야하며, 작을수록 조기 종료
EMPTY_PENALTY   = -0.15     # 탐색한 칸이 빈칸이면 보상 감소
DENSITY_COEF    = 0.15      # 탐색한 곳에 텍스트가 밀집되어 있을수록 더 큰 보상
TOKEN_DENSITY_MAX = 10      # 몇개의 텍스트까지 인정해줄지 한계치를 설정

SAL_COEF        = 0.12      # ocr 수행시, 밝고 대비 큰 맵쪽에 더 이끌리도록함. 값이 낮을수록 영향 감소
DIR_COEF        = 0.02      # 아래 방향으로 탐색을 유도 (사람의 시선은 위에서 아래방향)

KL_BETA_MAX     = 0.5      # 값이 높을수록 초기에 자유탐색 횟수 증가, 이후 집중되어 탐색하기 시작
KL_BETA         = 0.0

# ★ 텍스트-셀 힌트 설정
HINT_COEF       = 0.3            # 한 셀 방문당 보너스
HINT_STEPS      = 10             # 앞 10 스텝만 유효

# CUDA 미 사용시, 일반 CPU로 가동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed(SEED)

# ───── util ───────────────────────────────────────────────────────────────

# 메뉴 단어 목록들을 {단어: 정수 id}의 사전 형태로 생성해 반환 
def build_vocab_index(vocab: List[str]) -> Dict[str, int]:
    return {t:i for i,t in enumerate(vocab)}

# 전체 vocab 길이만큼 0으로 초기화 후, 토큰이 사전에 있으면 해당 인덱스를 1.0로 설정
# 예) [0, 1, 0, 0, 1....]   
def tokens_to_bow(tokens: List[str], vdx: Dict[str,int]):
    v = np.zeros(len(vdx), np.float32)
    for t in tokens:
        if t in vdx: v[vdx[t]] = 1.0
    return v

# 주어진 텍스트에 대해 중앙 좌표 반환
def bbox_center(bb):
    if isinstance(bb[0], (list, tuple)):
        x = sum(p[0] for p in bb)/4; y = sum(p[1] for p in bb)/4
    else:
        x = (bb[0]+bb[2])/2; y = (bb[1]+bb[3])/2
    return x,y

# 주어진 텍스트에 대해, 시야박스 내에 있는지 확인
def is_inside(bb, box):
    x,y = bbox_center(bb); x1,y1,x2,y2 = box
    return x1<=x<x2 and y1<=y<y2

# 원본 이미지 → OpenCV Spectral Residual 기법으로 시각적 주목도(8×8 ~ N×N) 맵 생성
# 이를 통해 밝고 대비가 큰 곳에 더 큰 보상이 들어와 그쪽으로 탐험 유도
def compute_saliency_map(img: Image.Image, N=8):
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    try:
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
        ok, smap = sal.computeSaliency(img_bgr)
        if not ok: 
            raise RuntimeError
        smap = cv2.resize(smap.astype(np.float32), (N,N), cv2.INTER_AREA)
    except Exception:
        smap = np.ones((N,N), np.float32)/(N*N)
    smap -= smap.min(); smap /= smap.max()+1e-6
    return smap

# 인간 시선이 화면 위쪽 중심에 몰린다는 가정으로 2D 가우시간 분포 생성
def default_human_prior(N=8, sigma=0.8):
    Y,X = np.mgrid[0:N,0:N]
    cx,cy = (N-1)/2, N*0.3
    dist = np.hypot((X-cx)/(N/2),(Y-cy)/(N/2))
    prior = np.exp(-(dist**2)/sigma); prior /= prior.sum()
    return prior

# 에피소드 번호(ep) → KL 가중치 반환. 500 ep까지 0, 이후 1000 ep 동안 선형 증가하여 KL_BETA_MAX에 도달.
# 위 human_prior을 초기엔 규제, 이후 점근적으로 강화해 사람 시선 구현
def curriculum_KL(ep):                 
    global KL_BETA
    frac = max(0.0, min(1.0, (ep - warmup) / duration))
    KL_BETA = KL_BETA_MAX * frac

# ───── 환경 ───────────────────────────────────────────────────────────────
class GazeKioskEnv:
    """8×8 grid, relative move, + OCR-hint early reward."""

    def __init__(self, img_path:str, menu:List[str], goal_seq:List[str], human_prior=None):
        self.image = Image.open(img_path).convert("RGB")
        self.W,self.H = self.image.size
        self.N = VISION_GRID_N
        self.max_steps = MAX_STEPS

        self.vdx = build_vocab_index(menu)


        print("[Env] OCR 수행 중…")
        # OCR
        self.ocr_txt, self.ocr_bb = check_ocr_box(
            self.image, display_img=False, output_bb_format='xyxy', use_paddleocr=True)

        print(f"OCR 결과: {len(self.ocr_txt)}개의 텍스트 요소 발견.")
        print("감지된 텍스트:")
        for i, (text, bbox) in enumerate(zip(self.ocr_txt, self.ocr_bb)):
            print(f"  {i+1}. '{text}' at {bbox}")

        # 텍스트 셀 힌트맵
        self.token_cells = np.zeros((self.N,self.N), bool)
        for bb in self.ocr_bb:
            x,y = bbox_center(bb)
            gx = min(int(x/(self.W/self.N)), self.N-1)
            gy = min(int(y/(self.H/self.N)), self.N-1)
            self.token_cells[gy,gx] = True
        self.hint_map = self.token_cells.astype(np.float32)   # ★
        self.total_token_cells = self.token_cells.sum()

        self.saliency_map = compute_saliency_map(self.image, self.N)
        self.human_prior  = human_prior if human_prior is not None else default_human_prior(self.N)

        self.set_goal_sequence(goal_seq)
        self.reset()

    # API ------------------------------------------------
    def set_goal_sequence(self, seq): self.goal_seq = list(seq)

    def reset(self):
        self.goal_idx = 0; self.steps = 0; self.done=False
        self.gx = self.N//2; self.gy = 0
        self.prev_gy = self.gy; self._update_pos()

        self.visited = np.zeros((self.N,self.N), int)
        self.edge_streak = -1
        return self._obs()

    def _update_pos(self):
        self.pos = [(self.gx+0.5)/self.N, (self.gy+0.5)/self.N]

    def step(self, a:int):
        dx,dy = MOVE_OPTIONS[a]
        self.gx = int(np.clip(self.gx+dx,0,self.N-1))
        self.gy = int(np.clip(self.gy+dy,0,self.N-1))
        self._update_pos(); self.steps+=1
        self.visited[self.gy,self.gx]+=1
        vcnt = self.visited[self.gy,self.gx]

        # ---------- 보상 ----------
        r = MOVE_COST
        r += DIST_PENALTY * max(0.0, math.hypot(dx,dy)-1)
        r += SAL_COEF * self.saliency_map[self.gy,self.gx]

        dir_y = self.gy - self.prev_gy
        if dir_y>0: r += DIR_COEF * (dir_y/(self.N-1))
        self.prev_gy = self.gy

        # ★ 힌트 보상 (초반 K스텝 한정)
        if self.steps <= HINT_STEPS and self.hint_map[self.gy,self.gx]:
            r += HINT_COEF

        in_view = [t for t,b in zip(self.ocr_txt,self.ocr_bb) if is_inside(b,self._vbox())]
        token_cnt = len(in_view)
        if token_cnt==0: r += EMPTY_PENALTY
        else: r += DENSITY_COEF * min(token_cnt,TOKEN_DENSITY_MAX)/TOKEN_DENSITY_MAX

        if self.token_cells[self.gy,self.gx]:
            r += EXPLORATION_BONUS if vcnt==1 else DUP_PENALTY/math.sqrt(vcnt)

        edge = self.gx in (0,self.N-1) or self.gy in (0,self.N-1)
        self.edge_streak = self.edge_streak+1 if edge else 0
        if edge: r += EDGE_PENALTY * min(self.edge_streak,EDGE_CAP)

        visited_token = np.logical_and(self.visited>0, self.token_cells).sum()
        unvisited_ratio = 1 - visited_token/max(1,self.total_token_cells)
        density_scalar = min(token_cnt,TOKEN_DENSITY_MAX)/TOKEN_DENSITY_MAX if token_cnt else 0
        r += COVERAGE_COEF * unvisited_ratio * density_scalar

        found=False
        if self.goal_idx < len(self.goal_seq):
            goal = self.goal_seq[self.goal_idx]
            if any(goal in t for t in in_view):
                r += SEMANTIC_REWARD + GOAL_REWARD
                self.goal_idx+=1; found=True

        # done
        done = (visited_token>=self.total_token_cells*COVER_END_THRESH) or \
               (self.goal_idx>=len(self.goal_seq)) or \
               (self.steps>=self.max_steps)
        if done:
            if KL_BETA>0:
                p=(self.visited.astype(float)+1e-6); p/=p.sum()
                q=self.human_prior+1e-6; q/=q.sum()
                r += -KL_BETA * float((p*np.log(p/q)).sum())

            if self.steps >= self.max_steps:
                r -= 5.0   # 에피소드 길어지면 손해
            self.done=True


        return self._obs(), float(r), self.done, {"found":found}

    # helpers -------------------------------------------------
    def _vbox(self):
        bw,bh = self.W/self.N, self.H/self.N
        cx,cy = int(self.pos[0]*self.W), int(self.pos[1]*self.H)
        return max(0,cx-bw/2), max(0,cy-bh/2), \
               min(self.W,cx+bw/2), min(self.H,cy+bh/2)

    def _obs(self):
        in_view=[t for t,b in zip(self.ocr_txt,self.ocr_bb) if is_inside(b,self._vbox())]
        sal_val = self.saliency_map[self.gy,self.gx]
        bow=tokens_to_bow(in_view,self.vdx)
        visited_bin=(self.visited>0).astype(np.float32).flatten()
        return np.concatenate([
            [self.goal_idx/max(1,len(self.goal_seq)), *self.pos, sal_val],
            bow, visited_bin]).astype(np.float32)

# ───── Actor-Critic ────────────────────────────────────────────────────────
class GazeActorCritic(nn.Module):
    def __init__(self, obs_dim, hid=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(obs_dim,hid),nn.ReLU(),
                               nn.Linear(hid,hid),nn.ReLU())
        self.actor=nn.Linear(hid,TOTAL_ACTIONS)
        self.critic=nn.Linear(hid,1)
    def forward(self,x):
        z=self.net(x); return self.actor(z), self.critic(z).squeeze(-1)

# ───── GAE ────────────────────────────────────────────────────────────────
def compute_gae(rew,val,gamma=0.99,lam=0.95):
    val=np.concatenate([val,[0.0]]); adv,gae=[],0
    for t in reversed(range(len(rew))):
        delta=rew[t]+gamma*val[t+1]-val[t]
        gae=delta+gamma*lam*gae; adv.insert(0,gae)
    return adv,(val[:-1]+adv).tolist()

# ───── 학습 루프 ───────────────────────────────────────────────────────────
def train():
    menu=['커피','주문','아이스크림','고구마','티라미수','메뉴','케이크','31',
          '망고케이크','애니멀파','레디팩','블록팩','듬뿍딸기케이크','음료',
          '파티용품','32000원','30000원','치즈','큐브','골라먹는27','리얼초코27']
    env=GazeKioskEnv("screen2.png",menu,["dummy"])
    
    present = [t for t in menu if any(t in o for o in env.ocr_txt)]
    if not present:         # OCR이 다 실패한 예외 상황 방어
        present = menu[:]   # 전체 메뉴에서 샘플
    sampler = lambda: random.sample(present, 5)

    obs_dim=env.reset().shape[0]
    agent=GazeActorCritic(obs_dim).to(device)
    opt=optim.Adam(agent.parameters(),2.5e-4)

    ep_ret,ep_len=[],[]
    TOTAL_EP=2000; batch=32; clip=0.2
    for ep in range(TOTAL_EP):
        curriculum_KL(ep)          # ★ KL curriculum
        env.set_goal_sequence(sampler()); obs=env.reset()
        buf=[]; ret_ep=0
        while True:
            logits,_=agent(torch.tensor(obs,dtype=torch.float32,device=device).unsqueeze(0))
            dist=torch.distributions.Categorical(logits=logits)
            act=dist.sample().item(); logp=dist.log_prob(torch.tensor(act,device=device)).item()
            nxt,r,done,_=env.step(act)
            buf.append((obs,act,r,logp)); obs=nxt; ret_ep+=r
            if done: break

        obs_a,act_a,rew_a,logp_a=map(list,zip(*buf))
        with torch.no_grad():
            val_a = agent(
                torch.tensor(obs_a, dtype=torch.float32, device=device)
            )[1].cpu().numpy()
        adv,ret=compute_gae(rew_a,val_a)
        obs_t=torch.tensor(np.stack(obs_a),dtype=torch.float32,device=device)
        act_t=torch.tensor(act_a,device=device)
        adv_t=torch.tensor(adv,dtype=torch.float32,device=device)
        ret_t=torch.tensor(ret,dtype=torch.float32,device=device)
        logp_old=torch.tensor(logp_a,dtype=torch.float32,device=device)

        for _ in range(6):
            idx=torch.randperm(len(obs_t))
            for i in range(0,len(idx),batch):
                mb=idx[i:i+batch]
                logits,val=agent(obs_t[mb])
                dist=torch.distributions.Categorical(logits=logits)
                logp=dist.log_prob(act_t[mb])
                ratio=torch.exp(logp-logp_old[mb])
                s1=ratio*adv_t[mb]; s2=torch.clamp(ratio,1-clip,1+clip)*adv_t[mb]
                pol_loss=-torch.min(s1,s2).mean()
                val_loss=(val-ret_t[mb]).pow(2).mean()
                ent=dist.entropy().mean()
                loss=pol_loss+0.5*val_loss-0.01*ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(),0.5); opt.step()

        ep_ret.append(ret_ep); ep_len.append(len(buf))
        if (ep+1)%50==0:
            print(f"[EP {ep+1}] avgR={np.mean(ep_ret[-100:]):.2f} "
                  f"avgLen={np.mean(ep_len[-100:]):.1f}")

    os.makedirs("omniparser/gaze_logs_v6", exist_ok=True)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.plot(ep_ret); plt.title("Return")
    plt.subplot(1,2,2); plt.plot(ep_len); plt.title("Length")
    plt.tight_layout(); plt.savefig("omniparser/gaze_logs_v6/learning_curve.png")
    torch.save(agent.state_dict(),"omniparser/gaze_ppo_v6.pt")
    print("saved gaze_ppo_v6.pt + curve.png")

if __name__=="__main__":
    train()