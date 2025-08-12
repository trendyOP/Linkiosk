import random, time, os
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
# 필요한 OCR 함수 import (아래 예시)
from utils.utils import check_ocr_box  # 꼭 본인 환경에 맞게 import
# ... (필요시 easyocr 등)

# ==== 0. Constants ======================================
VISION_GRID_N = 6
TOTAL_ACTIONS = 4      # 상/하/좌/우
MAX_STEPS = 30
MOVE_PENALTY = -0.01   # 스텝당 고정 비용 (값을 약간 줄임)
DUP_PENALTY  = -0.08   # 중복 방문 페널티
GOAL_REWARD  = 1.5     # 목표 달성 보상
# JUMP_PENALTY와 DIST_PENALTY는 아래 방식으로 대체됩니다.
POTENTIAL_REWARD_COEFF = 1.0  # << 새로 추가: 거리 기반 보상 계수
EXPLORATION_BONUS = 0.05

# ----------------------------------------------------------------------------
class KioskOCRViewportEnv:
    """OCR 결과(ocr_text, ocr_bbox) 기반 viewport 탐색 환경 (수정된 버전)"""
    def __init__(
        self,
        image_path: str,
        initial_menu_sequence: List[str], # 시작 시 사용할 메뉴 순서
        full_menu_list: List[str],        # << 모든 메뉴 후보 리스트
        vision_grid_n: int = VISION_GRID_N,
        step_size: float = 1/VISION_GRID_N,
        screen_size: Tuple[int, int] = (1920, 1080),
        max_steps: int = MAX_STEPS
    ):
        self.image = Image.open(image_path)
        self.screen_size = list(screen_size)
        self.vision_grid_n = vision_grid_n
        self.step_size = step_size
        self.max_steps = max_steps
        self.menu_sequence = initial_menu_sequence.copy()
        self.full_menu_list = full_menu_list.copy() # 전체 메뉴 리스트 저장

        print(f"\n이미지 처리 중: {image_path}")
        print("OCR 수행 중...")

        # --- OCR 수행 ---
        ocr_text, ocr_bbox = check_ocr_box(
            self.image, display_img=False, output_bb_format='xyxy', goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, use_paddleocr=True
        )
        self.ocr_text = ocr_text
        self.ocr_bbox = ocr_bbox
        
        print(f"OCR 결과: {len(ocr_text)}개의 텍스트 요소 발견.")
        print("감지된 텍스트:")
        for i, (text, bbox) in enumerate(zip(ocr_text, ocr_bbox)):
            print(f"  {i+1}. '{text}' at {bbox}")

        # << 핵심 >> 모든 가능한 메뉴에 대한 위치를 미리 계산
        self._precompute_goal_locations()
        self.reset()

    def select_action_with_mask(logits, epsilon=0.1):
        logits = logits.detach().cpu().numpy().flatten()
        probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
        if random.random() < epsilon:
            return np.random.choice(np.arange(TOTAL_ACTIONS))
        return np.random.choice(np.arange(TOTAL_ACTIONS), p=probs/probs.sum())


    def _precompute_goal_locations(self):
        """
        환경 초기화 시, 모든 목표 텍스트의 정규화된 중심 좌표를 미리 계산하여 저장합니다.
        """
        self.goal_locations = {}
        # 전체 메뉴 리스트를 사용하여 모든 후보의 위치를 저장
        for goal_text in set(self.full_menu_list):
            locations = []
            for text, bbox in zip(self.ocr_text, self.ocr_bbox):
                if text == goal_text:
                    if isinstance(bbox[0], (list, tuple)):
                        bx = sum(pt[0] for pt in bbox) / 4
                        by = sum(pt[1] for pt in bbox) / 4
                    else:
                        bx = (bbox[0] + bbox[2]) / 2
                        by = (bbox[1] + bbox[3]) / 2
                    
                    norm_bx = bx / self.screen_size[0]
                    norm_by = by / self.screen_size[1]
                    locations.append(np.array([norm_bx, norm_by]))
            self.goal_locations[goal_text] = locations

    def _get_distance_to_closest_goal(self, position: np.ndarray) -> float:
        """
        주어진 위치(정규화)에서 현재 목표까지의 가장 짧은 유클리드 거리를 반환합니다.
        """
        if self.goal_idx >= len(self.menu_sequence):
            return 0.0

        goal_text = self.menu_sequence[self.goal_idx]
        locations = self.goal_locations.get(goal_text, [])

        if not locations:
            return 1.42 # 화면 대각선 길이(sqrt(1^2+1^2)) 근사치, 페널티 방지용
        
        distances = [np.linalg.norm(position - loc) for loc in locations]
        return min(distances)

    def reset(self):
        self.goal_idx = 0
        self.done = False
        self.steps = 0
        self.stamp_position = [0.5, 0.5]
        self.visited = np.zeros((self.vision_grid_n, self.vision_grid_n), dtype=np.int32)
        return self._get_obs()

    def _get_vision_box(self, center):
        h, w = self.screen_size[1], self.screen_size[0]
        box_w = w / self.vision_grid_n
        box_h = h / self.vision_grid_n
        cx, cy = int(center[0] * w), int(center[1] * h)
        x1, y1 = max(0, cx - box_w / 2), max(0, cy - box_h / 2)
        x2, y2 = min(w, cx + box_w / 2), min(h, cy + box_h / 2)
        return (x1, y1, x2, y2)

    def _get_obs(self):
        return {
            "goal_idx": self.goal_idx,
            "stamp_position": np.array(self.stamp_position),
            "visited": self.visited.flatten().copy()
        }

    def step(self, action: int):
        if self.done:
            return self._get_obs(), 0.0, self.done, {}

        prev_pos = np.array(self.stamp_position)
        prev_dist_to_goal = self._get_distance_to_closest_goal(prev_pos)

        move = [0, 0]
        if action == 0: move[1] -= self.step_size
        elif action == 1: move[1] += self.step_size
        elif action == 2: move[0] -= self.step_size
        elif action == 3: move[0] += self.step_size

        new_pos_list = [
            np.clip(self.stamp_position[0] + move[0], 0, 1),
            np.clip(self.stamp_position[1] + move[1], 0, 1),
        ]
        new_pos = np.array(new_pos_list)
        
        actually_moved = not np.array_equal(new_pos, prev_pos)
        self.stamp_position = new_pos_list

        # --- 보상 계산 (새로운 방식) ---
        reward = MOVE_PENALTY 
        
        new_dist_to_goal = self._get_distance_to_closest_goal(new_pos)
        potential_reward = prev_dist_to_goal - new_dist_to_goal
        reward += POTENTIAL_REWARD_COEFF * potential_reward
        
        gx = np.clip(int(self.stamp_position[0] * self.vision_grid_n), 0, self.vision_grid_n - 1)
        gy = np.clip(int(self.stamp_position[1] * self.vision_grid_n), 0, self.vision_grid_n - 1)

        if self.visited[gx, gy] == 0:
            reward += EXPLORATION_BONUS * 4   # << 2~5로 증폭
        else:
            reward += DUP_PENALTY * 2  
        self.visited[gx, gy] += 1
        
        coverage = np.sum(self.visited > 0) / (self.vision_grid_n ** 2)
        if coverage > 0.8:
            reward += 0.5

        if not actually_moved: reward -= 0.04
        
        
        # --- 목표 발견 체크 ---
        found = False
        if self.goal_idx < len(self.menu_sequence):
            goal = self.menu_sequence[self.goal_idx]
            x1, y1, x2, y2 = self._get_vision_box(self.stamp_position)
            
            for text, bbox in zip(self.ocr_text, self.ocr_bbox):
                if isinstance(bbox[0], (list, tuple)):
                    bx, by = sum(pt[0] for pt in bbox)/4, sum(pt[1] for pt in bbox)/4
                else:
                    bx, by = (bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2
                
                if x1 <= bx < x2 and y1 <= by < y2 and goal == text:
                    found = True
                    break
            
            if found:
                reward += GOAL_REWARD
                self.goal_idx += 1
                self.visited[:] = 0

        self.steps += 1
        if self.goal_idx >= len(self.menu_sequence) or self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, {"found": found}

# =============================================================================
class ElderlyPersonaWrapper:
    def __init__(self, env, error_prob=0.20, delay_mean=0.40, delay_std=0.10, misclick_prob=0.12):
        self.env = env
        self.error_prob    = error_prob
        self.delay_mean    = delay_mean
        self.delay_std     = delay_std
        self.misclick_prob = misclick_prob

    def reset(self):
        return self.env.reset()

    def step(self, action: int):
        time.sleep(max(0.0, random.gauss(self.delay_mean, self.delay_std)))
        if random.random() < self.error_prob:
            action = random.randrange(TOTAL_ACTIONS)
        if random.random() < self.misclick_prob:
            action = random.randrange(TOTAL_ACTIONS)
        return self.env.step(action)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim:int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU())
        self.actor  = nn.Linear(64, TOTAL_ACTIONS)
        self.critic = nn.Linear(64, 1)
    def forward(self, obs):
        h = self.shared(obs)
        logits = self.actor(h)
        value  = self.critic(h)
        return logits, value

def obs_to_vec(obs):
    v = [obs["goal_idx"]/10, obs["stamp_position"][0], obs["stamp_position"][1]]
    v.extend(obs["visited"])
    return np.array(v, np.float32)

def select_action_with_mask(logits):
    logits = logits.detach().cpu().numpy().flatten()
    probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
    action = np.random.choice(np.arange(TOTAL_ACTIONS), p=probs/probs.sum())
    return action

def train_ppo(
    image_path,
    menu_list,
    goal_episodes=150,
    update_freq=32,
    gamma=0.98,
    lam=0.95,
    epochs=6,
    vision_grid_n=VISION_GRID_N,
    step_size=1/VISION_GRID_N,
    model_save_path="agppo_model.pt",
    log_dir="agppo_logs"
):
    def randomize_goal_sequence(menu_pool):
        k = 2
        menus = random.sample(menu_pool, k)
        if '아이스크림' in menus:
            menus.remove('아이스크림')
        if '진정한고구마아이스크' in menus:
            nenus.remove('진정한고구마아이스크')
        menus.insert(0, '아이스크림')
        menus.insert(1, '진정한고구마아이스크')
        return menus

    env = KioskOCRViewportEnv(
        image_path=image_path,
        initial_menu_sequence=randomize_goal_sequence(menu_list), # 시작 시퀀스
        full_menu_list=menu_list,                                 # << 모든 메뉴 후보 전달
        vision_grid_n=vision_grid_n,
        step_size=step_size
    )

    episode_rewards = []
    episode_lengths = []
    step_rewards = []
    memory = []

    env = ElderlyPersonaWrapper(env)
    obs = env.reset()
    total_rewards = []

    os.makedirs(log_dir, exist_ok=True)
    step_log_fp = open(os.path.join(log_dir, "step_log.txt"), "w", encoding="utf-8")

    obs_dim = len(obs_to_vec(obs))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic(obs_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    t = 0
    while t < goal_episodes:
        env.env.menu_sequence = randomize_goal_sequence(menu_list)
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_length = 0

        while not done:
            obs_vec = torch.tensor(obs_to_vec(obs), dtype=torch.float32).unsqueeze(0).to(device)
            logits, value = net(obs_vec)
            action = select_action_with_mask(logits)
            next_obs, reward, done, info = env.step(action)

            logline = (f"ep {t} | act: {action} | reward: {reward:.2f} | done: {done} | "
                       f"goal_idx: {next_obs['goal_idx']} | "
                       f"stamp_pos: {next_obs['stamp_position']} | visited: {np.sum(next_obs['visited'])}\n")
            print(logline.strip())
            step_log_fp.write(logline)
            step_log_fp.flush()

            step_rewards.append(reward)
            memory.append((obs_vec.cpu().numpy(), action, reward, value.item()))
            obs = next_obs

            ep_reward += reward
            ep_length += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        total_rewards.append(ep_reward)
        t += 1

        # PPO 업데이트
        if len(memory) >= update_freq:
            batch = memory[:update_freq]
            memory = memory[update_freq:]
            states = torch.tensor(np.vstack([b[0] for b in batch]), dtype=torch.float32).to(device)
            actions = torch.tensor([b[1] for b in batch], dtype=torch.int64).to(device)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(device)
            values = torch.tensor([b[3] for b in batch], dtype=torch.float32).to(device)
            # GAE advantage
            advantages = []
            gae = 0
            next_value = 0
            for r, v in zip(reversed(rewards), reversed(values)):
                delta = r + gamma * next_value - v
                gae = delta + gamma * lam * gae
                advantages.insert(0, gae)
                next_value = v
            returns = values + torch.tensor(advantages).to(device)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
            # PPO update
            for _ in range(epochs):
                logits, values_pred = net(states)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions)
                ratio = torch.exp(log_probs - log_probs.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (values_pred.squeeze() - returns).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    step_log_fp.close()
    print(f"Mean total reward: {np.mean(total_rewards):.2f}")

    # --- 모델 저장 ---
    torch.save(net.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # --- 학습 결과 시각화 ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(episode_rewards, label='Episode Reward')
    axs[0].set_ylabel("Total Reward")
    axs[0].set_xlabel("Episode")
    axs[0].legend()
    axs[1].plot(episode_lengths, label='Episode Length', color='orange')
    axs[1].set_ylabel("Length")
    axs[1].set_xlabel("Episode")
    axs[1].legend()
    plt.suptitle("Viewport AGPPO Training Log")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "training_plot.png"))
    plt.show()

    plt.figure()
    plt.plot(step_rewards, label="Step Reward", alpha=0.5)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Step Reward Progress")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "step_reward_plot.png"))
    plt.show()

    return net

if __name__ == "__main__":
    # 실제 OCR에 나오는 텍스트와 동일하게 작성!
    menu_list = [
        '메뉴',  '아이스크림', '진정한고구마', '진정한티라미수'
    ]
    train_ppo(
        image_path="screen2.png",
        menu_list=menu_list,
        model_save_path="omniparser/agppo_model.pt",
        log_dir="omniparser/agppo_logs"
    )