import cv2
import time
import torch
import numpy as np
import pygame
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
from run_omniparser_with_ppo_v6 import (
    GazeKioskEnv,
    GazeActorCritic,
    VISION_GRID_N,
    build_vocab_index,
    is_inside,
    device,
)

"""
Gaze Kiosk Visualizer — Refactored & Window‑size Fix
───────────────────────────────────────────────────
• 행동 라벨 9종 (상/하/좌/우/대각 4방 + 제자리)
• 공통 시각화 헬퍼: draw_korean, draw_banner, render_frame
• 최초 cv2 창 크기 자동 조정 (이미지 비율 유지, 최대 1200px)
"""

# ───── 경로·상수 ──────────────────────────────────────────
SCREEN_PATH = "screen2.png"
MODEL_PATH = "omniparser/gaze_ppo_v6.pt"
GOAL_TEXT = "고구마"
MENU_LIST = [
    "커피",
    "주문",
    "아이스크림",
    "고구마",
    "티라미수",
    "메뉴",
    "케이크",
    "31",
    "망고케이크",
    "애니멀파",
    "레디팩",
    "블록팩",
    "듬뿍딸기케이크",
    "음료",
    "파티용품",
    "32000원",
    "30000원",
    "치즈",
    "큐브",
    "골라먹는27",
    "리얼초코27",
]

TEST_TASKS = [["디저트", "아이스크림", "고구마"],["고구마", "커피", "아이스크림"],["고구마", "커피", "아이스크림"]]

MAX_EP_STEPS = 90
MAX_VISIT = 8  # heat‑map intensity cap
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
FONT_SIZE = 32
ACTION_LABELS = {
    0: "위로 이동",
    1: "아래로 이동",
    2: "오른쪽으로 이동", # 오른쪽으로 이동
    3: "오른쪽아래로 이동", # 오른쪽아래로 이동
    4: "왼쪽 위로 이동", 
    5: "왼쪽 아래로 이동", # 왼쪽 아래로 이동
    6: "왼쪽으로 이동",
    7: "오른쪽 위로 이동",
    8: "제자리",
}
# 확률 패널 전용 화살표 라벨
PROB_LABELS = {0: "↑", 1: "↓", 2: "→", 3: "↘", 4: "↖", 5: "↙", 6: "←", 7: "↗", 8: "■"}

# ───── 함수 정의 ──────────────────────────────────────────────
# OpenCV BGR 이미지 → 한글 텍스트가 들어간 BGR 이미지
def draw_korean(img_bgr: np.ndarray, text: str, org: tuple[int, int], color=(0, 255, 255)) -> np.ndarray:
    
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    draw.text(org, text, font=font, fill=color[::-1])  # PIL은 RGB
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

#왼쪽 상단 반투명 배너에 여러 줄 텍스트 표시
def draw_banner(img_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # 배너 크기 계산
    widths, heights = zip(*(font.getbbox(t)[2:] for t in lines))
    banner_w = max(widths) + 40
    banner_h = sum(heights) + 40

    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([120, 40, 120 + banner_w, 40 + banner_h], fill=(0, 0, 0, 150))
    img_pil = Image.alpha_composite(img_pil, overlay)
    draw = ImageDraw.Draw(img_pil)

    y = 60
    for line in lines:
        draw.text((140, y), line, font=font, fill=(255, 255, 255, 255))
        y += font.getbbox(line)[3]

    return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

def draw_prob_panel(img: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """행동 확률 막대패널 — 겹침 방지를 위한 동적 레이아웃"""
    n = len(probs)
    # 레이아웃 파라미터 (확대 & 간격 증가)
    bar_h = 20
    gap = 28                  # 막대 간격 (bar_h 포함)
    top_pad = 30
    side_pad = 50
    bottom_pad = 30

    panel_w = 450             # 고정 가로폭
    panel_h = top_pad + (n - 1) * gap + bar_h + bottom_pad

    # 좌하단 위치 (여백 30 유지)
    x0, y0 = 120, img.shape[0] - panel_h - 150

    # 반투명 배경
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, dst=img)

    # 막대 시작 좌표
    bar_x = x0 + side_pad
    bar_y = y0 + top_pad
    bar_w = panel_w - side_pad * 2 - 90  # 오른쪽 텍스트 영역 90px 확보

    # OpenCV 막대 그리기
    for i, p in enumerate(probs):
        bar_len = int(bar_w * p)
        y1 = bar_y + i * gap
        cv2.rectangle(img, (bar_x, y1), (bar_x + bar_len, y1 + bar_h), (0, 255, 0), -1)

    # PIL 텍스트 그리기 (유니코드 지원)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    for i, p in enumerate(probs):
        label = PROB_LABELS.get(i, str(i))
        text = f"{label}: {p:.2f}"
        txt_x = bar_x + bar_w + 10
        txt_y = bar_y + i * gap + (bar_h - font.size) // 2
        draw.text((txt_x, txt_y), text, font=font, fill=(255, 255, 255, 255))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def render_frame(base: np.ndarray, visited: np.ndarray, vbox: tuple[int, int, int, int], seq, env, tid, step, act, found, center, probs,stars=[]):
    """히트맵 + 뷰포트 + 배너 종합 렌더"""
    H, W = base.shape[:2]
    cell_w, cell_h = W // VISION_GRID_N, H // VISION_GRID_N

    show = base.copy()
    overlay = show.copy()
    for sx, sy in stars:
        cv2.putText(show, "o", (sx-18, sy+18), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 4, cv2.LINE_AA)

    # 방문 셀마다 빨간색 오버레이 (셀별로 alpha 다르게)
    for gy in range(VISION_GRID_N):
        for gx in range(VISION_GRID_N):
            v = visited[gy, gx]
            if v == 0:
                continue
            alpha = min(v / MAX_VISIT, 1.0)  # 방문 많이 할수록 더 불투명 (최대 1)
            cx1, cy1 = gx * cell_w, gy * cell_h
            cx2, cy2 = cx1 + cell_w, cy1 + cell_h
            # ROI 슬라이싱
            roi = show[cy1:cy2, cx1:cx2]
            red_rect = np.full_like(roi, (0,0,255))  # 빨간색(BGR)
            # alpha blending
            show[cy1:cy2, cx1:cx2] = cv2.addWeighted(red_rect, alpha, roi, 1 - alpha, 0)

    # 뷰포트 박스
    x1, y1, x2, y2 = vbox
    color = (0, 0, 255) if found else (0, 255, 0)
    cv2.rectangle(show, (x1, y1), (x2, y2), color, 3)

    goal_txt = seq[min(env.goal_idx, len(seq) - 1)]
    banner_lines = [
        f"Step {step}",
        f"목표: {GOAL_TEXT}",
        f"행동: {ACTION_LABELS.get(act, '?')}",
        f"상태: {'목표 발견' if found else '탐색중...'}",
    ]
    if found:
        banner_lines.append(f"클릭: {center}")
    show = draw_banner(show, banner_lines)
    show = draw_prob_panel(show, probs)
    return show


# ───── 메인 루프 ─────────────────────────────────────────

def main():
    global GOAL_TEXT
    env = GazeKioskEnv(SCREEN_PATH, MENU_LIST, ["dummy"])
    env.max_steps = MAX_EP_STEPS
    vocab = build_vocab_index(MENU_LIST)
    obs_dim = env.reset().shape[0]

    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=256)
    click_snd = pygame.mixer.Sound("click.wav") 

    # 모델 로드
    net = GazeActorCritic(obs_dim).to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()

    # 기본 이미지 및 창 설정
    base_img = cv2.imread(SCREEN_PATH)
    H, W = base_img.shape[:2]

    cv2.namedWindow("Test", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    # 초기 창 크기 조정 (너비 1200px 이하, 비율 유지)
    disp_w = 1920 #min(W, 1200)
    disp_h = 1080 #int(H * disp_w / W)
    cv2.resizeWindow("Test", disp_w, disp_h)

    stars = []
    first = True

    for tid, seq in enumerate(TEST_TASKS, 1):
        print(f"\n== TASK {tid}: {seq}")
        env.set_goal_sequence(seq)
        obs = env.reset()
        visited = np.zeros((VISION_GRID_N, VISION_GRID_N), np.uint8)
        step = 0
        if first:
            cv2.destroyWindow("Test")
            first=False
        while True:
            step += 1
            with torch.no_grad():
                logits, _ = net(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            act = torch.distributions.Categorical(logits=logits).sample().item()
            probs = F.softmax(logits.squeeze(), dim=0).cpu().numpy()

            obs, _, done, info = env.step(act)
            visited[env.gy, env.gx] = min(visited[env.gy, env.gx] + 1, MAX_VISIT)

            x1, y1, x2, y2 = map(int, env._vbox())
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            show = render_frame(
                base_img,
                visited,
                (x1, y1, x2, y2),
                seq,
                env,
                tid,
                step,
                act,
                info.get("found", False),
                center,
                probs,
                stars,
            )
            cv2.imshow("Test", show)

            if info.get("found", False):
                stars.append(center)

            print(
                f"step {step:2d} act={act} ({ACTION_LABELS.get(act)}) gx,gy=({env.gx},{env.gy}) "
                f"goal='{seq[min(env.goal_idx,len(seq)-1)]}' found={info.get('found')}"
            )

            GOAL_TEXT = seq[min(env.goal_idx,len(seq)-1)]

        

             # 키 처리 & 프레임 간 지연
            if info.get("found"):
                click_snd.play()
                if cv2.waitKey(1000) == 27:
                    return  # ESC → quit
            else:
                if cv2.waitKey(600) == 27:
                    return  # ESC → quit
            
            
        

            # 에피소드 종료 done or 
            if step>=MAX_EP_STEPS:
                stars = []
                if env.goal_idx == len(seq):
                    print("  >> ALL GOALS FOUND")
                else:
                    print("  >> Episode ended (timeout or coverage)")
                break
        time.sleep(0.6)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()