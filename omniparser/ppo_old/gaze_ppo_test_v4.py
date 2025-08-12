# gaze_ppo_test_v4.py  (FIXED dtype)
# -------------------------------------------------------
import cv2, difflib, numpy as np, torch, time
from PIL import Image
from utils.utils import check_ocr_box
from run_omniparser_with_ppo_v4 import (
    GazeActorCritic, build_vocab_index, tokens_to_bow,
    is_inside, VISION_GRID_N, STEP_SIZE, device
)


SCREEN_PATH = "screen2.png"
MODEL_PATH  = "omniparser/gaze_ppo_v4.pt"
MENU_LIST   = ['커피','주문','아이스크림','진정한고구마','진정한티라미수','메뉴']

TEST_TASKS = [
    ["커피", "아이스크림","망고케이크", "메뉴"],
    ["아이스크림","고구마","아이스크림", "고구마", "메뉴"],
    ["아이스크림","메뉴","진정한치즈아이스크림"],
]

def goal_found(goal, txts, th=.85):
    g = goal.replace(' ','').lower()
    return any(
        g in t.replace(' ','').lower() or
        difflib.SequenceMatcher(None, g, t.replace(' ','').lower()).ratio() >= th
        for t in txts )

def vbox(pos, n=VISION_GRID_N, scr=(1920,1080)):
    W, H   = scr
    bw, bh = W // n, H // n           # 셀 크기
    gx = min(int(pos[0] * n), n - 1)  # 셀 인덱스
    gy = min(int(pos[1] * n), n - 1)
    x1, y1 = gx * bw, gy * bh
    x2, y2 = x1 + bw, y1 + bh
    return x1, y1, x2, y2, gx, gy


def main():
    img = Image.open(SCREEN_PATH); bg = np.array(img)
    H, W = bg.shape[:2]                    # 화면 크기
    ocr_txt, ocr_bb = check_ocr_box(img, display_img=False,
                                    output_bb_format='xyxy', use_paddleocr=True)

    vocab = build_vocab_index(MENU_LIST)
    obs_dim = 74    # v4 환경 기준: 1(goal) + 2(pos) + 1(density) + 6(bow) + 64(visited)

    net = GazeActorCritic(obs_dim).to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()

    cv2.namedWindow("Test", cv2.WINDOW_NORMAL)

    MAX_VISIT = 15                         # 5번 이상은 동일한 진한 빨강
    cell_w, cell_h = W // VISION_GRID_N, H // VISION_GRID_N

    for tid, seq in enumerate(TEST_TASKS, 1):
        print(f"\n== TASK {tid}: {seq}")
        pos = [0, 0]
        visited = np.zeros((VISION_GRID_N, VISION_GRID_N), dtype=np.uint8)  # 누적 방문수
        goal_i = 0

        for step in range(1000):
            x1, y1, x2, y2, gx, gy = vbox(pos, scr=(W, H))
            in_view = [t for t,b in zip(ocr_txt, ocr_bb) if is_inside(b,(x1,y1,x2,y2))]
            bow     = tokens_to_bow(in_view, vocab)
            token_cnt = len(in_view)
            density_scalar = min(token_cnt, 10) / 10.0     # TOKEN_DENSITY_MAX 동일

            obs = np.concatenate([
                [goal_i / len(seq), *pos, density_scalar],  # ← density_scalar 추가
                bow,
                (visited > 0).astype(np.float32).flatten()
            ]).astype(np.float32)

            with torch.no_grad():
                logits, _ = net(torch.tensor(obs, device=device).unsqueeze(0))
            act = torch.distributions.Categorical(logits=logits).sample().item()

            # 위치 업데이트
            dx,dy = [(0,-1),(0,1),(-1,0),(1,0)][act]
            pos[0] = float(np.clip(pos[0] + dx*STEP_SIZE, 0, 1))
            pos[1] = float(np.clip(pos[1] + dy*STEP_SIZE, 0, 1))

            visited[gy, gx] = np.clip(visited[gy, gx] + 1, 0, MAX_VISIT)

            goal  = seq[goal_i]
            found = goal_found(goal, in_view)
            print(f"step {step+1:2d} pos=({pos[0]:.2f},{pos[1]:.2f}) "
                  f"act={act} goal='{goal}' found={found}")

            # ── 시각화 ───────────────────────────────────────────────
            show    = bg.copy()
            overlay = show.copy()           # 열에 칠할 임시 캔버스

            for yi in range(VISION_GRID_N):
                for xi in range(VISION_GRID_N):
                    vcnt = visited[yi, xi]
                    if vcnt == 0:           # 방문 안한 칸 패스
                        continue
                    intensity = int(255 * vcnt / MAX_VISIT)   # 1→연한  …  5→진한
                    cx1 = xi * cell_w
                    cy1 = yi * cell_h
                    cx2 = cx1 + cell_w
                    cy2 = cy1 + cell_h
                    cv2.rectangle(overlay, (cx1,cy1), (cx2,cy2),
                                   color=(0,0,intensity), thickness=-1)  # 빨강 채우기

            # 빨간 칸을 알파‑블렌딩하여 반투명 효과
            cv2.addWeighted(overlay, 0.55, show, 0.45, 0, show)

            # 현재 시야 박스
            cv2.rectangle(show, (x1,y1), (x2,y2),
                          (0,0,255) if found else (0,255,0), 3)
            cv2.putText(show, f"T{tid} S{step+1}", (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(show, f"Goal: {goal}", (30,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,100,0), 2)

            cv2.imshow("Test", show)
            if cv2.waitKey(150) == 27: return     # ESC → 종료

            if found:
                goal_i += 1
                # visited[:] = 0      목표 달성 후 초기화
                if goal_i == len(seq):
                    print("  >> ALL GOALS FOUND")
                    break
        time.sleep(.5)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()