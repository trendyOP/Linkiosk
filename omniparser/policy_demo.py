

import time
import cv2
import numpy as np
import pygame
import torch
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw
import os
import platform

# local imports from refactored agent
from run_omniparser_with_ppo_v9 import (
    GazeKioskEnv,
    GazeActorCritic,
    MOVE_OPTIONS,
    VISION_GRID_N,
    RC,  # RewardConfig instance
    device,
)

# ────────────────────────────────
# Config paths & constants
# ────────────────────────────────

SCREEN_PATH  = "screen5.png"                       # sample kiosk screenshot
MODEL_PATH   = "omniparser/gaze_ppo_v9.pt"       # trained agent weights

MAX_EP_STEPS = RC.max_steps
MAX_VISIT    = 10    # heat‑map intensity cap
FONT_SIZE    = 28
OBS_DIM      = 214   # 모델 입력 차원을 상수로 정의

# 크로스 플랫폼 폰트 경로
def get_font_path():
    system = platform.system()
    if system == "Windows":
        return r"C:\Windows\Fonts\malgun.ttf"
    elif system == "Darwin":  # macOS
        return "/System/Library/Fonts/Arial.ttf"
    else:  # Linux
        return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

FONT_PATH = get_font_path()

ACTION_LABELS = {
    0: "↑", 1: "↗", 2: "→", 3: "↘", 4: "↓", 5: "↙", 6: "←", 7: "↖", 8: "■"
}

# ────────────────────────────────
# Drawing helpers
# ────────────────────────────────

def _font():
    try:
        return ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except OSError:
        # 폰트 로드 실패 시 기본 폰트 사용
        return ImageFont.load_default()


def draw_banner(img_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(img_pil)
    font = _font()

    # banner dims
    widths, heights = zip(*(font.getbbox(t)[2:] for t in lines))
    bw, bh = max(widths) + 40, sum(heights) + 40
    x0, y0 = 40, 30

    # semi‑transparent bg
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle([x0, y0, x0 + bw, y0 + bh], fill=(0, 0, 0, 160))
    img_pil = Image.alpha_composite(img_pil, overlay)
    draw = ImageDraw.Draw(img_pil)

    y = y0 + 20
    for ln in lines:
        draw.text((x0 + 20, y), ln, font=font, fill=(255, 255, 255, 255))
        y += font.getbbox(ln)[3]

    return cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def draw_prob_panel(img: np.ndarray, probs: np.ndarray) -> np.ndarray:
    bar_h, gap = 18, 26
    top_pad, side_pad, bottom_pad = 20, 40, 20
    n = len(probs)
    panel_w = 420
    panel_h = top_pad + n * gap + bottom_pad

    H, W = img.shape[:2]
    x0, y0 = W - panel_w - 40, 40

    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, dst=img)

    bar_x = x0 + side_pad
    bar_w = panel_w - side_pad * 2 - 80

    for i, p in enumerate(probs):
        y1 = y0 + top_pad + i * gap
        cv2.rectangle(img, (bar_x, y1), (bar_x + int(bar_w * p), y1 + bar_h), (0, 255, 0), -1)

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = _font()
    for i, p in enumerate(probs):
        txt = f"{ACTION_LABELS[i]} {p:.2f}"
        draw.text((bar_x + bar_w + 10, y0 + top_pad + i * gap - 4), txt, font=font, fill=(255, 255, 255, 255))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def render_frame(base: np.ndarray, visited: np.ndarray, vbox: tuple[int, int, int, int],
                 step: int, act: int, cov: float, probs: np.ndarray) -> np.ndarray:
    """Overlay heat‑map, viewport and HUD onto the kiosk screenshot."""
    H, W = base.shape[:2]
    cw, ch = W // VISION_GRID_N, H // VISION_GRID_N

    show = base.copy()

    # heat‑map overlay
    for gy in range(VISION_GRID_N):
        for gx in range(VISION_GRID_N):
            v = visited[gy, gx]
            if v == 0:
                continue
            alpha = min(v / MAX_VISIT, 1.0)
            x1, y1 = gx * cw, gy * ch
            roi = show[y1:y1 + ch, x1:x1 + cw]
            red = np.full_like(roi, (0, 0, 255))
            show[y1:y1 + ch, x1:x1 + cw] = cv2.addWeighted(red, alpha, roi, 1 - alpha, 0)

    # viewport rectangle
    cv2.rectangle(show, (vbox[0], vbox[1]), (vbox[2], vbox[3]), (0, 255, 0), 3)

    # HUD banner
    banner = [f"Step {step}", f"Coverage: {cov*100:.1f}%", f"Action: {ACTION_LABELS[act]}"]
    show = draw_banner(show, banner)
    show = draw_prob_panel(show, probs)
    return show

# ────────────────────────────────
# Main test loop
# ────────────────────────────────

def main():
    # 파일 존재 확인
    if not os.path.exists(SCREEN_PATH):
        print(f"Error: Screen image not found at {SCREEN_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    env = GazeKioskEnv(img_path=SCREEN_PATH, verbose=True)
    env.max_steps = MAX_EP_STEPS  # just to be explicit

    # 모델 로드
    try:
        agent = GazeActorCritic(OBS_DIM).to(device)
        agent.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        agent.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 관찰 벡터를 모델 차원에 맞게 조정하는 함수
    def adjust_obs_dim(obs, target_dim=OBS_DIM):
        current_dim = obs.shape[0]
        if current_dim == target_dim:
            return obs
        elif current_dim > target_dim:
            # 차원이 크면 앞쪽부터 자르기
            return obs[:target_dim]
        else:
            # 차원이 작으면 0으로 패딩
            padded = np.zeros(target_dim, dtype=obs.dtype)
            padded[:current_dim] = obs
            return padded

    base_img = cv2.imread(SCREEN_PATH)
    if base_img is None:
        print(f"Error: Could not load image from {SCREEN_PATH}")
        return
        
    visited = np.zeros((VISION_GRID_N, VISION_GRID_N), np.uint8)

    cv2.namedWindow("ScanTest", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow("ScanTest", 1280, 720)

    episode = 0
    while True:
        episode += 1
        obs = env.reset()
        visited.fill(0)
        step = 0
        print(f"\n== EPISODE {episode} ==")

        while True:
            step += 1
            # 관찰 벡터를 모델 차원에 맞게 조정
            adjusted_obs = adjust_obs_dim(obs)
            with torch.no_grad():
                logits, _ = agent(torch.tensor(adjusted_obs, dtype=torch.float32, device=device).unsqueeze(0))
            probs = F.softmax(logits.squeeze(), dim=0).cpu().numpy()
            act = torch.distributions.Categorical(logits=logits).sample().item()

            obs, _, done, info = env.step(act)
            visited[env.gy, env.gx] = min(visited[env.gy, env.gx] + 1, MAX_VISIT)

            x1, y1, x2, y2 = map(int, env._vbox())
            frame = render_frame(base_img, visited, (x1, y1, x2, y2), step, act, cov=info["grid_cov"], probs=probs)
            cv2.imshow("ScanTest", frame)

            key = cv2.waitKey(20)
            if key == 27:
                cv2.destroyAllWindows()
                return  # ESC quits

            if done:
                print(f" episode ends: coverage={info['grid_cov']*100:.1f}%")
                cv2.waitKey(800)  # brief pause before next run
                break


if __name__ == "__main__":
    main()