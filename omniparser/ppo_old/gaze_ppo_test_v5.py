# gaze_ppo_test_v5.py  (Saliency + Jump‑Action 환경용)
# -------------------------------------------------------------
"""
Test script for the *gaze_ppo_saliency_v5* policy.

변경 핵심
─────────
* **64‑action jump** : act ∈ [0, 63] → (gx, gy) 셀 중심으로 즉시 이동
* **STEP_SIZE 상수 제거** – 셀 중심 좌표 직접 계산
* 모델 체크포인트 `gaze_ppo_v5.pt` 로드
"""
import cv2, difflib, numpy as np, torch, time
from PIL import Image

from run_omniparser_with_ppo_v5 import (
    GazeActorCritic, build_vocab_index, tokens_to_bow,
    is_inside, VISION_GRID_N, device
)

# ───── 경로·상수 ─────────────────────────────────────────────
SCREEN_PATH = "screen2.png"                      # 테스트용 스크린샷
MODEL_PATH  = "omniparser/gaze_ppo_v5.pt"       # v5 체크포인트
MENU_LIST = ['커피', '주문', '아이스크림', '고구마', '티라미수', 
            '메뉴', '케이크', '31', '망고케이크', '애니멀파', '레디팩', '블록팩', 
            '듬뿍딸기케이크', '음료', '파티용 품', '32000원', '30000원', '치즈', '큐브', '골라먹는27', '리얼초코27']

TEST_TASKS = [
    ["커피", "아이스크림","망고케이크", "메뉴"],
    ["아이스크림","고구마","아이스크림", "고구마", "메뉴"],
    ["아이스크림","메뉴","애니멀파"],
    ['못찾는거']
]

TOKEN_DENSITY_MAX = 10       # v5와 동일
vocab   = build_vocab_index(MENU_LIST)
obs_dim = 4 + len(vocab) + VISION_GRID_N**2      
MAX_STEPS         = 1000

# ───── 헬퍼 ─────────────────────────────────────────────────

def goal_found(goal, txts, th: float = 0.85):
    g = goal.replace(' ', '').lower()
    return any(
        g in t.replace(' ', '').lower() or
        difflib.SequenceMatcher(None, g, t.replace(' ', '').lower()).ratio() >= th
        for t in txts
    )


def center_pos_from_action(act: int, n: int = VISION_GRID_N):
    """act → normalized (x,y) at cell center."""
    gx = act % n              # 열
    gy = act // n             # 행
    return (gx + 0.5) / n, (gy + 0.5) / n


def vbox(pos, n: int = VISION_GRID_N, scr=(1920, 1080)):
    """Return bounding box (pixel) for current viewport cell."""
    W, H = scr
    bw, bh = W // n, H // n
    gx = min(int(pos[0] * n), n - 1)
    gy = min(int(pos[1] * n), n - 1)
    x1, y1 = gx * bw, gy * bh
    x2, y2 = x1 + bw, y1 + bh
    return x1, y1, x2, y2, gx, gy


# ───── 메인 루프 ────────────────────────────────────────────

def main():
    # 이미지 & OCR 사전 계산
    img = Image.open(SCREEN_PATH)
    bg  = np.array(img)
    H, W = bg.shape[:2]

    from utils.utils import check_ocr_box  # 지연 import (paddleocr GPU 점유 최소화)
    ocr_txt, ocr_bb = check_ocr_box(img, display_img=False,
                                    output_bb_format='xyxy', use_paddleocr=True)

    vocab = build_vocab_index(MENU_LIST)

    # 모델 로드
    net = GazeActorCritic(obs_dim).to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()

    cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
    MAX_VISIT = 10
    cell_w, cell_h = W // VISION_GRID_N, H // VISION_GRID_N

    for tid, seq in enumerate(TEST_TASKS, 1):
        print(f"\n== TASK {tid}: {seq}")
        pos = [0.5, 0.5]            # 시작은 화면 중앙
        visited = np.zeros((VISION_GRID_N, VISION_GRID_N), np.uint8)
        goal_i = 0

        for step in range(MAX_STEPS):
            # 현재 셀 파라미터
            x1, y1, x2, y2, gx, gy = vbox(pos, scr=(W, H))
            in_view = [t for t, b in zip(ocr_txt, ocr_bb) if is_inside(b, (x1, y1, x2, y2))]
            bow = tokens_to_bow(in_view, vocab)
            density_scalar = min(len(in_view), TOKEN_DENSITY_MAX) / TOKEN_DENSITY_MAX

            obs = np.concatenate([
                [goal_i / len(seq), *pos, density_scalar],
                bow,
                (visited > 0).astype(np.float32).flatten()
            ]).astype(np.float32)

            with torch.no_grad():
                logits, _ = net(torch.tensor(obs, device=device).unsqueeze(0))
            act = torch.distributions.Categorical(logits=logits).sample().item()

            # 위치 업데이트 (jump)
            pos[0], pos[1] = center_pos_from_action(act)
            # 방문 카운트 업데이트
            _, _, _, _, gx, gy = vbox(pos, scr=(W, H))
            visited[gy, gx] = min(visited[gy, gx] + 1, MAX_VISIT)

            goal = seq[goal_i]
            found = goal_found(goal, in_view)
            print(f"step {step+1:2d} pos=({pos[0]:.2f},{pos[1]:.2f}) act={act} goal='{goal}' found={found}")

            # ── 시각화 ────────────────────────────────────
            show = bg.copy()
            overlay = show.copy()
            for yi in range(VISION_GRID_N):
                for xi in range(VISION_GRID_N):
                    vcnt = visited[yi, xi]
                    if vcnt == 0:
                        continue
                    intensity = int(255 * vcnt / MAX_VISIT)
                    cx1, cy1 = xi * cell_w, yi * cell_h
                    cx2, cy2 = cx1 + cell_w, cy1 + cell_h
                    cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), (0, 0, intensity), -1)
            cv2.addWeighted(overlay, 0.55, show, 0.45, 0, show)
            cv2.rectangle(show, (x1, y1), (x2, y2), (0, 0, 255) if found else (0, 255, 0), 3)
            cv2.putText(show, f"T{tid} S{step+1}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(show, f"Goal: {goal}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
            cv2.imshow("Test", show)
            if found:
                if cv2.waitKey(400) == 27:
                    return  # ESC → quit
            elif len(in_view)>0 :
                if cv2.waitKey(250) == 27:
                    return  # ESC → quit
            else:
                if cv2.waitKey(40) == 27:
                    return  # ESC → quit

            if found:
                goal_i += 1
                # visited[:] = 0  # 목표 달성 후 방문 초기화 필요 시 사용
                if goal_i == len(seq):
                    print("  >> ALL GOALS FOUND")
                    break
        time.sleep(0.5)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
