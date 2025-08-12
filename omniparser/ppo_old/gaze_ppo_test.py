# gaze_ppo_test_v3_heatmap.py
# -------------------------------------------------------
"""Gaze‑based PPO test with heat‑map visualization.
Shows cumulative attention by tinting frequently viewed grid cells red.
"""

import cv2
import time
import difflib
import numpy as np
import torch
from PIL import Image

from utils.utils import check_ocr_box
from run_omniparser_with_ppo_v2 import (
    GazeActorCritic,
    build_vocab_index,
    tokens_to_bow,
    is_inside,
    VISION_GRID_N,
    STEP_SIZE,
    device,
)

# ─── Paths & parameters ─────────────────────────────────────────────────────────
SCREEN_PATH = "screen2.png"
MODEL_PATH = "omniparser/gaze_ppo_v3.pt"
SCREEN_SIZE = (1920, 1080)
MENU_LIST = ["커피", "주문", "아이스크림", "진정한고구마", "진정한티라미수", "메뉴"]

TEST_TASKS = [
    ["아이스크림", "케이크", "딸기케이크"],
    ["아이스크림", "네모블록미키마우스", "진정한치즈아이스크림"],
    ["아이스크림", "망고케이크", "커피"],
]

# ─── Similarity / partial‑match utility ─────────────────────────────────────────

def goal_found(goal: str, texts, th: float = 0.85) -> bool:
    g = goal.replace(" ", "").lower()
    for t in texts:
        tn = t.replace(" ", "").lower()
        if g in tn or difflib.SequenceMatcher(None, g, tn).ratio() >= th:
            return True
    return False


# ─── Vision box utility ---------------------------------------------------------

def vision_box_xyxy(pos, n=VISION_GRID_N, screen=SCREEN_SIZE):
    W, H = screen
    bw, bh = W // n, H // n
    cx, cy = int(pos[0] * W), int(pos[1] * H)
    return max(0, cx - bw // 2), max(0, cy - bh // 2), min(W, cx + bw // 2), min(H, cy + bh // 2)


# ─── Main loop ──────────────────────────────────────────────────────────────────

def main():
    # Load screenshot (force RGB → BGR to guarantee 3 channels)
    img_pil = Image.open(SCREEN_PATH).convert("RGB")  # drops alpha if any
    bg_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # OCR: get texts & bounding boxes
    ocr_txt, ocr_bb = check_ocr_box(img_pil, display_img=False, output_bb_format="xyxy", use_paddleocr=True)
    print(f"OCR 요소 {len(ocr_txt)}개")

    # Model
    vocab_idx = build_vocab_index(MENU_LIST)
    obs_dim = 2 + len(MENU_LIST) + VISION_GRID_N * VISION_GRID_N + 1
    net = GazeActorCritic(obs_dim).to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()

    # Heat‑map arrays
    visit_cnt = np.zeros((VISION_GRID_N, VISION_GRID_N), np.int32)
    VIS_MAX = 20  # ≥20 visits = fully red

    cv2.namedWindow("Gaze‑PPO Test", cv2.WINDOW_NORMAL)

    for tid, seq in enumerate(TEST_TASKS, 1):
        print(f"\n==== [TASK {tid}] {seq} ====")
        goal_idx, pos = 0, [0.5, 0.5]
        visited = np.zeros((VISION_GRID_N, VISION_GRID_N), np.int32)

        for step in range(30):
            # Vision box & texts inside
            x1, y1, x2, y2 = vision_box_xyxy(pos)
            in_view = [t for t, b in zip(ocr_txt, ocr_bb) if is_inside(b, (x1, y1, x2, y2))]
            bow = tokens_to_bow(in_view, vocab_idx)

            # Observation vector (excluding scalar goal_idx)
            obs_vec = np.concatenate(
                [
                    np.array([goal_idx / len(seq), pos[0], pos[1]], np.float32),
                    bow,
                    visited.astype(np.float32).flatten(),
                ]
            )
            logits, _ = net(torch.tensor(obs_vec).to(device))
            action = torch.argmax(logits).item()

            # Move gaze
            dx = [0, 0, -STEP_SIZE, STEP_SIZE][action] if action > 1 else 0
            dy = [-STEP_SIZE, STEP_SIZE, 0, 0][action] if action < 2 else 0
            pos[0] = float(np.clip(pos[0] + dx, 0, 1))
            pos[1] = float(np.clip(pos[1] + dy, 0, 1))

            gx, gy = min(int(pos[0] * VISION_GRID_N), VISION_GRID_N - 1), min(
                int(pos[1] * VISION_GRID_N), VISION_GRID_N - 1
            )
            visited[gy, gx] = 1
            visit_cnt[gy, gx] += 1

            # Goal logic
            goal = seq[goal_idx]
            found = goal_found(goal, in_view)

    
            print(
                f"step {step + 1:2d} pos=({pos[0]:.2f},{pos[1]:.2f}) action={action} "
                f"goal='{goal}' found={found}"
            )

            # ---------------------------------------------------------------------
            # Compose frame with heat‑map overlay
            show = bg_img.copy()

            # Build heat image (same size & channels as bg_img)
            heat = np.zeros_like(show, np.uint8)
            cell_w, cell_h = show.shape[1] // VISION_GRID_N, show.shape[0] // VISION_GRID_N

            for iy in range(VISION_GRID_N):
                for ix in range(VISION_GRID_N):
                    c = visit_cnt[iy, ix]
                    if c == 0:
                        continue
                    alpha = min(1.0, c / VIS_MAX)
                    intensity = int(alpha * 255)
                    cx1, cy1 = ix * cell_w, iy * cell_h
                    cx2, cy2 = cx1 + cell_w, cy1 + cell_h
                    cv2.rectangle(heat, (cx1, cy1), (cx2, cy2), (0, 0, intensity), -1)

            show = cv2.addWeighted(show, 1.0, heat, 0.4, 0)

            # Draw current vision box & HUD
            cv2.rectangle(show, (x1, y1), (x2, y2), (0, 0, 255) if found else (0, 255, 0), 3)
            cv2.putText(show, f"Task {tid} Step {step + 1}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(show, f"Goal: {goal}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)

            cv2.imshow("Gaze‑PPO Test", show)
            if cv2.waitKey(200) == 27:  # Esc
                return

            # Next goal?
            if found:
                print(f"  >>> FOUND {goal}")
                goal_idx += 1
                visited[:] = 0
                if goal_idx >= len(seq):
                    print("  >>> ALL TARGETS ACHIEVED")
                    break
        time.sleep(0.3)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()