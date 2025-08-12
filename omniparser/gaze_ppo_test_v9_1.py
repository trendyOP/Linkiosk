import cv2
import time
import torch
import numpy as np
import pygame
import pyautogui
import mss
import types
import difflib
from dataclasses import dataclass
from collections import defaultdict
import torch.nn.functional as F
from PIL import Image
from text_normalizer import normalize_token

from utils.utils import check_ocr_box
# ▼ v8 → v9
from run_omniparser_with_ppo_v9 import (
    GazeKioskEnv,
    GazeActorCritic,
    VISION_GRID_N,
    build_vocab_index,
    is_inside,
    device,
    MOVE_OPTIONS,
    compute_saliency_map,
    bbox_center,
)


# ───── 경로·상수 ──────────────────────────────────────────
SCREEN_PATH = "screen5.png"               # 초기 더미 이미지
# ▼ v8 모델 경로 → v9
MODEL_PATH  = "omniparser/gaze_ppo_v9.pt"
MENU_LIST = [
    "커피", "주문", "아이스크림", "고구마", "티라미수", "메뉴", "케이크", "31", "망고케이크",
    "애니멀파", "레디팩", "블록팩", "듬뿍딸기케이크", "음료", "파티용품", "32000원", "30000원",
    "치즈", "큐브", "골라먹는27", "리얼초코27",
]
TEST_TASKS = [["아메리카노","ICE전용","주문담기","더담기", "카푸치노", "ICE전용", "주문담기", "결제하기", "확인","신용카드","예"]]
MAX_EP_STEPS = 1000

# 헬퍼
from typing import List
def _set_goal_sequence(self, seq: List[str]):
    """목표 토큰 시퀀스를 환경 인스턴스에 기록하고 인덱스를 초기화"""
    self._goal_seq = seq
    self.goal_idx  = 0

def _goal_step_postprocess(self, in_view: List[str]):
    """
    • current viewport(input: in_view)에 목표 토큰이 보이면 True 반환
    • self.goal_idx 를 1 증가 (다음 목표로 이동)
    • self._last_candidates : viewport 안 모든 매칭 bb 저장
    """
    found = False
    self._last_candidates = []
    if self.goal_idx < len(self._goal_seq):
        tgt = self._goal_seq[self.goal_idx]
        for txt, bb in zip(self.ocr_txt, self.ocr_bb):
            if txt and tgt in txt and is_inside(bb, self._vbox()):  # ✅ viewport 내부만
                self._last_candidates.append((txt, bb))
        if self._last_candidates:
            self.goal_idx += 1
            found = True
    return found


# ─── Memory Manager ───────────────────────────────────────
@dataclass
class Mem:
    gx: int; gy: int; step: int; quality: float

class MemoryManager:
    def __init__(self, cap=400, ttl=2000):
        self.store = defaultdict(list); self.cap, self.ttl = cap, ttl

    def _qual(self, dist_center: float, step: int) -> float:
        return (1.0 - dist_center) * 0.7 + (step / 1000.0) * 0.3

        
    def update(self, tokens, gx, gy, step, dist_center):
        """메모리에 토큰 좌표·품질 기록
        ‣ key  : normalize_token(t)  → ‘ICE전용’·‘|CE’·‘ice’ 모두 같은 슬롯
        ‣ value: Mem(gx, gy, step, quality) 는 그대로 유지
        """
        q = self._qual(dist_center, step)

        for t_raw in tokens:
            t = normalize_token(t_raw)            # ★ 1) 정규화한 문자열로 key 통일
            if not t:                     # 빈 문자열이면 건너뜀(‘절대모살…’ 방지)
                continue

            lst = self.store[t]                   # 동일 key 에 좌표들 누적
            idx = next((i for i, m in enumerate(lst)
                        if m.gx == gx and m.gy == gy), None)

            if idx is not None:                   # 이미 있으면 step·quality 갱신
                lst[idx].step    = step
                lst[idx].quality = max(lst[idx].quality, q)
            else:                                 # 처음 보는 좌표
                lst.append(Mem(gx, gy, step, q))

            lst.sort(key=lambda m: (-m.quality, -m.step))
            del lst[self.cap:]   

    def prune(self, cur_step):
        for tok in list(self.store.keys()):
            self.store[tok] = [m for m in self.store[tok] if cur_step - m.step <= self.ttl]
            if not self.store[tok]: del self.store[tok]

        
    def best(self, token: str, min_ratio: float = 0.80):
        tok_n = normalize_token(token)

        # ① 정규화 후 완전 일치 우선
        for k in self.store.keys():
            if tok_n == normalize_token(k):
                return self.store[k][0]

        # ② 유사도 비교 (정규화 버전)
        best_key, best_score = None, 0.0
        for k in self.store.keys():
            score = difflib.SequenceMatcher(None, tok_n, normalize_token(k)).ratio()
            if score > best_score:
                best_key, best_score = k, score

        if best_key is not None and best_score >= min_ratio:
            return self.store[best_key][0]
        return None

# 이동방향 → action id 매핑
MOVE2IDX = {d: i for i, d in enumerate(MOVE_OPTIONS)}

def plan_move(curr, tgt):
    dx = 0 if curr[0] == tgt[0] else (1 if tgt[0] > curr[0] else -1)
    dy = 0 if curr[1] == tgt[1] else (1 if tgt[1] > curr[1] else -1)
    return MOVE2IDX.get((dx, dy), 0)

# ─── 화면 갱신 메서드 (OCR 재실행) ─────────────────────────

def _refresh_screen(self: GazeKioskEnv, bgr_img: np.ndarray):
    """실제 스크린샷을 주입해 OCR·saliency·token 셀을 다시 계산"""
    self.image = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    self.W, self.H = self.image.size
    self.cell_w, self.cell_h = self.W / self.N, self.H / self.N

    self.ocr_txt, self.ocr_bb = check_ocr_box(
        self.image, display_img=False, output_bb_format="xyxy", use_paddleocr=True
    )

    self.saliency_map = compute_saliency_map(self.image, self.N)
    self.token_cells.fill(False)
    for bb in self.ocr_bb:
        x, y = bbox_center(bb)
        gx = min(int(x / self.cell_w), self.N - 1)
        gy = min(int(y / self.cell_h), self.N - 1)
        self.token_cells[gy, gx] = True
    self.hint_map = self.token_cells.astype(np.float32)
    self.total_token_cells = self.token_cells.sum()

    return self._obs()

# ────── 메인 루프 ─────────────────────────────────────────

def main():
    #, use_appraisal=True
    env = GazeKioskEnv(SCREEN_PATH, verbose=False) 
    env.max_steps = MAX_EP_STEPS
    env._refresh_screen = types.MethodType(_refresh_screen, env)  # 바인딩(1회)

    env.set_goal_sequence    = types.MethodType(_set_goal_sequence, env)
    env._goal_step_postprocess = types.MethodType(_goal_step_postprocess, env)

    obs_dim = env.reset().shape[0]
    net = GazeActorCritic(obs_dim).to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device)); net.eval()

    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=256)
    click_snd = pygame.mixer.Sound("click.wav")

    mon = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # 실제 창 좌표로 조정
    sct = mss.mss()

    memory = MemoryManager(); seek_mode = False

    for tid, seq in enumerate(TEST_TASKS, 1):
        print(f"\n== TASK {tid}: {seq}")
        env.set_goal_sequence(seq); obs = env.reset(); step = 0
        need_refresh = True          # ← OCR 갱신 플래그
        while True:
            step += 1
            print("step : ", step)

            # ① 필요한 경우에만 캡처 & OCR
            if need_refresh:
                scr = np.array(sct.grab(mon))[:, :, :3]
                obs = env._refresh_screen(scr)
                cv2.imwrite("debug_capture.png", scr)
                print("OCR boxes:", len(env.ocr_txt), env.ocr_txt[:])
                need_refresh = False
                print("mem:", {k:[(m.gx,m.gy) for m in v] for k,v in memory.store.items()})

            # ② 메모리 타깃 선정
            goal_tok = seq[env.goal_idx] if env.goal_idx < len(seq) else None
            mem_target = memory.best(goal_tok) if goal_tok else None

            if mem_target:
                seek_mode = True
                act = plan_move((env.gx, env.gy), (mem_target.gx, mem_target.gy))
            else:
                seek_mode = False
                with torch.no_grad():
                    logits, _ = net(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                act = torch.distributions.Categorical(logits=logits).sample().item()

            # ③ 환경 스텝 진행
            obs, _, done, info = env.step(act)

            # ③-1 viewport 따라 마우스 이동
            PADDING = 5                                 # 모서리 여유
            vx1, vy1, vx2, vy2 = env._vbox()
            cx_raw = (vx1 + vx2) / 2
            cy_raw = (vy1 + vy2) / 2

            if (step%3==0):
                x = int(cx_raw) + mon["left"]
                y = int(cy_raw) + mon["top"]
                x = max(PADDING, min(x, mon["left"] + mon["width"]  - 1 - PADDING))
                y = max(PADDING, min(y, mon["top"]  + mon["height"] - 1 - PADDING))

                pyautogui.moveTo(x, y, duration=0.1)
                

            # ④ 메모리 업데이트
            in_view = [t for t, b in zip(env.ocr_txt, env.ocr_bb) if is_inside(b, env._vbox())]
            found = env._goal_step_postprocess(in_view)

            cx, cy = (env._vbox()[0] + env._vbox()[2]) // 2, (env._vbox()[1] + env._vbox()[3]) // 2
            dist_center = ((cx - env.W / 2) / env.W) ** 2 + ((cy - env.H / 2) / env.H) ** 2
            memory.update(in_view, env.gx, env.gy, step, dist_center ** 0.5); memory.prune(step)

            # ⑤ 목표 찾으면 실제 클릭 + OCR 갱신 예약
            if found: 
                # 1) step() 에서 저장한 우선순위 후보
                cand = env._last_candidates
                if cand:                                # exact / prefix ≤ 2 글자만 들어 있음
                    _, bb = cand[0]
                    cx_raw, cy_raw = bbox_center(bb)
                else:                                   # 혹시라도 후보가 비어 있으면
                    cx_raw, cy_raw = bbox_center(env._vbox())

                click_x = int(cx_raw) + mon["left"]
                click_y = int(cy_raw) + mon["top"]

                pyautogui.moveTo(click_x, click_y, duration=0.15)
                pyautogui.click()
                click_snd.play()
                print(f"Clicked at {(click_x, click_y)} for token '{goal_tok}'")

                obs = env.reset_scan() 
                time.sleep(0.3)          # UI 전환 대기
                
                need_refresh = True
                seek_mode = False
                continue
            
            
            # ⑥ 종료 조건
            if step >= MAX_EP_STEPS or env.goal_idx == len(seq):
                msg = "ALL GOALS FOUND" if env.goal_idx == len(seq) else "Episode ended (timeout)"
                print("mem:", {k:[(m.gx,m.gy) for m in v] for k,v in memory.store.items()})
                print("  >>", msg); break


    print("\n✔ 모든 TASK 종료")

if __name__ == "__main__":
    main()