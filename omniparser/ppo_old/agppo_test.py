import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from utils.utils import check_ocr_box  # 네 환경에 맞게 import
from run_omniparser_with_ppo import ActorCritic, obs_to_vec  # 네가 훈련할 때 쓰던 거 import (함수/클래스 이름 일치시켜야 함!)

# --- 환경 및 PPO 모델 세팅 ---
SCREEN_PATH = "screen2.png"
MODEL_PATH = "omniparser/agppo_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VISION_GRID_N = 6
STEP_SIZE = 1/VISION_GRID_N
SCREEN_SIZE = (1920, 1080)

# --- 테스트할 메뉴 시퀀스 (3개) ---
TEST_TASKS = [
    ["아이스크림", "케이크", "더듬뿍딸기케이크"],
    ["아이스크림", "네모블록미키마우스", "진정한치즈아이스크림"],
    ["아이스크림", "더듬뿍망고케이크", "진정한고구마아이스크"]
]

def draw_vision_box(img, norm_pos, grid_n, color=(0,255,0), thickness=4):
    h, w = img.shape[:2]
    box_w, box_h = w // grid_n, h // grid_n
    cx, cy = int(norm_pos[0]*w), int(norm_pos[1]*h)
    x1 = max(0, cx - box_w//2)
    y1 = max(0, cy - box_h//2)
    x2 = min(w, cx + box_w//2)
    y2 = min(h, cy + box_h//2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return (x1, y1, x2, y2)

def main():
    # 1. 이미지 및 OCR
    image = Image.open(SCREEN_PATH)
    np_img = np.array(image)
    ocr_text, ocr_bbox = check_ocr_box(image, display_img=False, output_bb_format='xyxy', use_paddleocr=True)
    print(f"OCR 결과: {len(ocr_text)}개 텍스트")
    for i, (txt, bb) in enumerate(zip(ocr_text, ocr_bbox)):
        print(f" {i+1}. '{txt}' at {bb}")

    # 2. PPO 모델 로드
    obs_dim = 2 + 1 + VISION_GRID_N*VISION_GRID_N  # goal_idx, stamp x/y, visited grid
    net = ActorCritic(obs_dim).to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.eval()

    for task_id, menu_sequence in enumerate(TEST_TASKS):
        print(f"\n==== [테스트 TASK {task_id+1}] 메뉴 시퀀스: {menu_sequence} ====")
        goal_idx = 0
        done = False
        stamp_position = [0.5, 0.5]
        visited = np.zeros((VISION_GRID_N, VISION_GRID_N), dtype=np.int32)
        max_steps = 30
        step = 0

        cv2.namedWindow("AGPPO Test", cv2.WINDOW_NORMAL)
        while not done and step < max_steps:
            # obs 만들기
            obs = {
                "goal_idx": goal_idx,
                "stamp_position": np.array(stamp_position),
                "visited": visited.flatten().copy()
            }
            obs_vec = torch.tensor(obs_to_vec(obs), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits, value = net(obs_vec)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
            action = np.argmax(probs)

            # 이동
            prev_pos = stamp_position.copy()
            move = [0, 0]
            if action == 0: move[1] -= STEP_SIZE
            elif action == 1: move[1] += STEP_SIZE
            elif action == 2: move[0] -= STEP_SIZE
            elif action == 3: move[0] += STEP_SIZE
            stamp_position[0] = float(np.clip(stamp_position[0] + move[0], 0, 1))
            stamp_position[1] = float(np.clip(stamp_position[1] + move[1], 0, 1))

            gx = int(stamp_position[0] * VISION_GRID_N)
            gy = int(stamp_position[1] * VISION_GRID_N)
            gx = np.clip(gx, 0, VISION_GRID_N-1)
            gy = np.clip(gy, 0, VISION_GRID_N-1)
            visited[gx, gy] = 1

            # --- goal 체크 (OCR 기반) ---
            found = False
            goal = menu_sequence[goal_idx]
            x1, y1, x2, y2 = draw_vision_box(np_img.copy(), stamp_position, VISION_GRID_N)  # 시각화용
            menus_in_vision = []
            goal_center = None
            for text, bbox in zip(ocr_text, ocr_bbox):
                if isinstance(bbox[0], (list, tuple)):
                    bx = sum([pt[0] for pt in bbox]) / 4
                    by = sum([pt[1] for pt in bbox]) / 4
                else:
                    bx = (bbox[0] + bbox[2]) // 2
                    by = (bbox[1] + bbox[3]) // 2
                if x1 <= bx < x2 and y1 <= by < y2:
                    menus_in_vision.append(text)
                    if goal == text:
                        found = True
                        goal_center = (bx / SCREEN_SIZE[0], by / SCREEN_SIZE[1])
            # 시각화
            show_img = np.array(image).copy()
            draw_vision_box(show_img, stamp_position, VISION_GRID_N, color=(0,0,255) if found else (0,255,0), thickness=4)
            cv2.putText(show_img, f"Task {task_id+1} / Step {step+1}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 2)
            cv2.putText(show_img, f"Goal: {goal}", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,100,0), 2)
            cv2.imshow("AGPPO Test", show_img)
            cv2.waitKey(400)  # 0.4초 대기

            print(f"step {step+1}: pos={stamp_position}, action={action}, goal={goal}, found={found}")

            if found:
                print(f"  >>> [발견] {goal}")
                goal_idx += 1
                visited[:] = 0
                if goal_idx >= len(menu_sequence):
                    print("  >>> [모든 목표 메뉴 달성]")
                    done = True
            step += 1

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
