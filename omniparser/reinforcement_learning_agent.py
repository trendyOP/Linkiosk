import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime

class ElderlyMemory:
    """고령자 특성을 시뮬레이션하는 메모리 시스템"""
    
    def __init__(self, max_items=3, forget_steps=5, memory_decay_rate=0.8):
        self.memory = []
        self.max_items = max_items
        self.forget_steps = forget_steps
        self.memory_decay_rate = memory_decay_rate
        
    def see(self, item):
        """새로운 정보를 기억"""
        # 기존에 같은 아이템이 있는지 확인
        for m in self.memory:
            if m['item'] == item:
                m['steps'] = 0  # 기억 갱신
                return
        
        # 새로운 정보 추가
        self.memory.append({'item': item, 'steps': 0, 'confidence': 1.0})
        
        # 메모리 용량 초과 시 가장 오래된 정보 삭제
        if len(self.memory) > self.max_items:
            self.memory.pop(0)
    
    def step(self):
        """한 스텝마다 기억력 감소"""
        for m in self.memory:
            m['steps'] += 1
            # 기억 신뢰도 감소
            m['confidence'] *= self.memory_decay_rate
        
        # 일정 스텝이 지나면 잊어버림
        self.memory = [m for m in self.memory if m['steps'] < self.forget_steps]
    
    def recall(self, item):
        """기억에서 아이템을 찾기"""
        for m in self.memory:
            if m['item'] == item:
                return m['confidence']
        return 0.0
    
    def get_memory_state(self):
        """현재 메모리 상태 반환"""
        return {
            'items': [m['item'] for m in self.memory],
            'confidences': [m['confidence'] for m in self.memory],
            'steps': [m['steps'] for m in self.memory]
        }

class ElderlyRLAgent(nn.Module):
    """고령자 특성을 반영한 강화학습 에이전트"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ElderlyRLAgent, self).__init__()
        
        # Actor 네트워크 (정책)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic 네트워크 (가치 함수)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 고령자 메모리 시스템
        self.memory = ElderlyMemory()
        
        # 학습 파라미터
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.entropy_coef = 0.01
        
        # 옵티마이저
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # 행동 히스토리
        self.action_history = []
        self.reward_history = []
        self.memory_history = []
        
    def forward(self, state):
        """상태를 받아서 행동 확률과 가치를 반환"""
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value
    
    def get_state_vector(self, stamp_position, detected_buttons, ocr_text, vision_grid, 
                        elderly_info=None):
        """상태를 벡터로 변환 (훈련 데이터와 동일한 구조)"""
        state_vector = []
        
        # 1. 시선 위치 (2차원) - 스탬프 위치를 시선 위치로 사용
        x_norm = stamp_position[0] if stamp_position[0] <= 1.0 else 0.5
        y_norm = stamp_position[1] if stamp_position[1] <= 1.0 else 0.5
        state_vector.extend([x_norm, y_norm])
        
        # 2. 시선 신뢰도 (1차원) - 기본값 0.8 (실제 시선추적에서는 confidence 값)
        confidence = 0.8
        state_vector.append(confidence)
        
        # 3. 시나리오 진행률 (1차원) - 현재 단계 / 전체 단계
        # 실행 시에는 시나리오 정보가 없으므로 기본값 0
        progress = 0.0
        state_vector.append(progress)
        
        # 4. 시선 속도 (1차원) - 현재 시선 이동 속도
        velocity = elderly_info.get('gaze_velocity', 0.0) if elderly_info else 0.0
        state_vector.append(velocity)
        
        # 5. 시선 가속도 (1차원) - 현재 시선 이동 가속도
        acceleration = elderly_info.get('gaze_acceleration', 0.0) if elderly_info else 0.0
        state_vector.append(acceleration)
        
        # 6. 최근 클릭과의 시간 차이 (1차원)
        time_diff = elderly_info.get('reaction_time', 0.0) if elderly_info else 0.0
        state_vector.append(time_diff)
        
        # 7. 시선 고정 시간 (1차원) - 현재 고정 시간
        fixation_duration = elderly_info.get('fixation_duration', 0.0) if elderly_info else 0.0
        state_vector.append(fixation_duration)
        
        # 8. 시나리오 단계별 정보 (10차원) - 현재 단계와 목표 메뉴 정보
        step_info = [0.0] * 10
        # 실행 시에는 시나리오 정보가 없으므로 모두 0
        state_vector.extend(step_info)
        
        # 9. 메모리 상태 (2차원) - 고령자 메모리 시스템
        memory_state = self.memory.get_memory_state()
        memory_items = len(memory_state['items'])
        memory_age = memory_state.get('age', 0)
        state_vector.extend([memory_items, memory_age])
        
        # 10. 패딩 (130차원) - 150차원으로 맞춤
        padding = [0.0] * 130
        state_vector.extend(padding)
        
        return torch.FloatTensor(state_vector)
    
    def select_action(self, state_vector):
        """상태를 받아서 행동을 선택"""
        action_probs, value = self.forward(state_vector)
        
        # 행동 분포 생성
        dist = Categorical(action_probs)
        action = dist.sample()
        
        return action.item(), action_probs, value, dist.entropy()
    
    def update_memory(self, detected_buttons, ocr_text):
        """메모리 업데이트"""
        # 감지된 버튼들을 메모리에 저장
        for button in detected_buttons:
            if 'text' in button:
                self.memory.see(button['text'])
        
        # OCR 텍스트를 메모리에 저장
        if ocr_text:
            for text in ocr_text:
                self.memory.see(text)
    
    def calculate_elderly_reward(self, action, detected_buttons, target_menu, 
                                stamp_position, reaction_time, gaze_velocity, ocr_text=None, ocr_bbox=None):
        """고령자 특성을 반영한 보상 계산"""
        reward = 0.0
        
        # 1. 목표 메뉴 찾기 보상
        for button in detected_buttons:
            if 'text' in button and target_menu in button['text']:
                reward += 15.0  # 목표 찾기 성공 (보상 증가)
                break
        
        # 2. 메모리 활용 보상
        memory_recall = 0.0
        for button in detected_buttons:
            if 'text' in button:
                recall = self.memory.recall(button['text'])
                memory_recall = max(memory_recall, recall)
        
        if memory_recall > 0.5:
            reward += 5.0  # 기억 활용 성공
        elif memory_recall < 0.1:
            reward -= 2.0  # 기억 활용 실패
        
        # 3. 반응 속도 보상 (훈련 데이터 특성 반영)
        if reaction_time > 1.5:  # 1.5초 이상 (적당히 느림)
            reward += 2.0  # 자연스러운 반응
        elif reaction_time < 0.3:  # 너무 빠름
            reward -= 2.0  # 비현실적
        
        # 4. 시선 이동 속도 보상 (훈련 데이터 특성 반영)
        if gaze_velocity > 600:  # 너무 빠른 시선 이동
            reward -= 1.0
        elif 100 <= gaze_velocity <= 400:  # 적당한 속도 (훈련 데이터 범위)
            reward += 3.0  # 자연스러운 시선 이동
        elif gaze_velocity < 50:  # 너무 느린 시선 이동
            reward -= 1.0
        
        # 5. 반복 탐색 패널티
        if len(self.action_history) > 3:
            recent_actions = self.action_history[-3:]
            if len(set(recent_actions)) == 1:  # 같은 행동 반복
                reward -= 3.0
        
        # 6. 텍스트/이미지가 있는 곳을 보는 것에 대한 보상 (강화)
        min_distance = float('inf')
        target_found = False
        closest_target_type = None
        
        # 버튼과의 거리 계산
        for button in detected_buttons:
            if 'bbox' in button:
                bbox = button['bbox']
                button_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                dist = np.sqrt((stamp_position[0] - button_center[0])**2 + 
                             (stamp_position[1] - button_center[1])**2)
                if dist < min_distance:
                    min_distance = dist
                    closest_target_type = 'button'
                target_found = True
        
        # OCR 텍스트와의 거리도 계산
        if ocr_text and ocr_bbox:
            for i, (text, bbox) in enumerate(zip(ocr_text, ocr_bbox)):
                try:
                    # bbox가 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 형태인 경우
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        if isinstance(bbox[0], list):
                            # 4개 점의 평균을 중심으로 계산
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            center_x = sum(x_coords) / len(x_coords)
                            center_y = sum(y_coords) / len(y_coords)
                        else:
                            # 기존 [x1, y1, x2, y2] 형태
                            center_x = (bbox[0] + bbox[2]) / 2
                            center_y = (bbox[1] + bbox[3]) / 2
                        
                        dist = np.sqrt((stamp_position[0] - center_x)**2 + 
                                     (stamp_position[1] - center_y)**2)
                        if dist < min_distance:
                            min_distance = dist
                            closest_target_type = 'ocr_text'
                        target_found = True
                except (TypeError, IndexError):
                    continue
        
        # 텍스트/이미지가 있는 곳을 보는 것에 보상 (강화된 보상)
        if target_found:
            if min_distance < 0.05:  # 매우 가까이 있음
                reward += 10.0  # 높은 보상
                if closest_target_type == 'ocr_text':
                    reward += 5.0  # OCR 텍스트에 추가 보상
            elif min_distance < 0.1:  # 가까이 있음
                reward += 7.0
                if closest_target_type == 'ocr_text':
                    reward += 3.0
            elif min_distance < 0.2:  # 적당한 거리
                reward += 4.0
                if closest_target_type == 'ocr_text':
                    reward += 2.0
            elif min_distance < 0.3:  # 멀지 않은 거리
                reward += 1.0
            elif min_distance > 0.5:  # 멀리 있음
                reward -= 3.0  # 패널티 증가
        else:
            reward -= 2.0  # 텍스트/이미지가 없는 곳을 보면 패널티 증가
        
        return reward
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """한 스텝 학습"""
        states = torch.stack(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.BoolTensor(dones)
        
        # 현재 상태의 행동 확률과 가치
        action_probs, values = self.forward(states)
        
        # 다음 상태의 가치
        _, next_values = self.forward(next_states)
        
        # Advantage 계산
        advantages = rewards + self.gamma * next_values * (~dones).float() - values
        
        # Actor 손실 (정책 그래디언트)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic 손실
        critic_loss = advantages.pow(2).mean()
        
        # 엔트로피 정규화
        entropy = dist.entropy().mean()
        
        # 전체 손실
        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
        
        # 역전파
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }

class ElderlyRLTrainer:
    """고령자 특성 강화학습 훈련기"""
    
    def __init__(self, agent, data_dir):
        self.agent = agent
        self.data_dir = data_dir
        self.training_data = []
        
    def load_training_data(self):
        """수집된 UIUX 인증 데이터 로드"""
        print("훈련 데이터 로드 중...")
        
        # 데이터 디렉토리 탐색
        for root, dirs, files in os.walk(self.data_dir):
            for dir_name in dirs:
                if dir_name.startswith('uiux_certification_'):
                    data_path = os.path.join(root, dir_name)
                    
                    # 시나리오 데이터 로드
                    scenario_file = os.path.join(data_path, 'scenario_data.json')
                    if os.path.exists(scenario_file):
                        with open(scenario_file, 'r', encoding='utf-8') as f:
                            scenario_data = json.load(f)
                        
                        # 시선 데이터 로드
                        gaze_file = os.path.join(data_path, 'gaze_data.json')
                        if os.path.exists(gaze_file):
                            with open(gaze_file, 'r', encoding='utf-8') as f:
                                gaze_data = json.load(f)
                        
                        # 클릭 이벤트 로드
                        click_file = os.path.join(data_path, 'click_events.json')
                        if os.path.exists(click_file):
                            with open(click_file, 'r', encoding='utf-8') as f:
                                click_events = json.load(f)
                        
                        # 타임라인 데이터 로드
                        timeline_file = os.path.join(data_path, 'timeline_data.json')
                        if os.path.exists(timeline_file):
                            with open(timeline_file, 'r', encoding='utf-8') as f:
                                timeline_data = json.load(f)
                        
                        # 훈련 데이터로 변환
                        episode_data = self.convert_to_training_data(
                            scenario_data, gaze_data, click_events, timeline_data
                        )
                        self.training_data.append(episode_data)
        
        print(f"총 {len(self.training_data)}개의 에피소드 데이터 로드 완료")
    
    def convert_to_training_data(self, scenario_data, gaze_data, click_events, timeline_data):
        """실제 데이터를 강화학습 훈련 데이터로 변환"""
        episode = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': []
        }
        
        # 시선 데이터를 기반으로 상태 시퀀스 생성
        for i, gaze_point in enumerate(gaze_data):
            # 상태 생성 (시선 위치, 감지된 버튼, OCR 텍스트 등)
            state = self.create_state_from_gaze(gaze_point, scenario_data)
            
            # 행동 생성 (시선 이동 방향)
            action = self.create_action_from_gaze(gaze_point, gaze_data, i)
            
            # 보상 생성 (고령자 특성 반영)
            reward = self.create_elderly_reward(gaze_point, click_events, scenario_data)
            
            # 다음 상태
            next_state = state  # 간단히 같은 상태로 설정
            
            # 종료 조건
            done = i == len(gaze_data) - 1
            
            episode['states'].append(state)
            episode['actions'].append(action)
            episode['rewards'].append(reward)
            episode['next_states'].append(next_state)
            episode['dones'].append(done)
        
        return episode
    
    def create_state_from_gaze(self, gaze_point, scenario_data):
        """시선 데이터로부터 상태 벡터 생성"""
        # 간단한 상태 벡터 생성
        state_vector = [
            gaze_point['x'] / 1920,  # 정규화된 x 좌표
            gaze_point['y'] / 1080,  # 정규화된 y 좌표
            gaze_point.get('confidence', 0.5),  # 시선 추적 신뢰도
            scenario_data.get('current_step', 0) / len(scenario_data.get('scenario_steps', [])),  # 진행률
        ]
        
        return torch.FloatTensor(state_vector)
    
    def create_action_from_gaze(self, gaze_point, gaze_data, index):
        """시선 이동으로부터 행동 생성"""
        if index == 0:
            return 0  # 첫 번째 포인트는 정지
        
        # 이전 시선과의 차이로 이동 방향 결정
        prev_gaze = gaze_data[index - 1]
        dx = gaze_point['x'] - prev_gaze['x']
        dy = gaze_point['y'] - prev_gaze['y']
        
        # 4방향 중 가장 가까운 방향 선택
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2  # 우/좌
        else:
            return 1 if dy > 0 else 0  # 하/상
    
    def create_elderly_reward(self, gaze_point, click_events, scenario_data):
        """고령자 특성을 반영한 보상 생성"""
        reward = 0.0
        
        # 클릭 이벤트와의 시간 차이
        for click_event in click_events:
            time_diff = abs(gaze_point['timestamp'] - click_event['timestamp'])
            if time_diff < 1.0:  # 1초 이내
                reward += 5.0  # 클릭 전후 시선 보상
        
        # 시선 고정 시간 보상
        if 'velocity' in gaze_point and gaze_point['velocity'] < 100:
            reward += 1.0  # 천천히 움직이는 것에 보상
        
        # 시선 신뢰도 보상
        if 'confidence' in gaze_point:
            if gaze_point['confidence'] > 0.8:
                reward += 2.0
            elif gaze_point['confidence'] < 0.3:
                reward -= 1.0
        
        return reward
    
    def train(self, num_episodes=1000, batch_size=32):
        """강화학습 훈련"""
        print("강화학습 훈련 시작...")
        
        training_history = {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropies': []
        }
        
        for episode in range(num_episodes):
            # 랜덤하게 에피소드 선택
            if self.training_data:
                episode_data = np.random.choice(self.training_data)
                
                # 배치 크기만큼 데이터 샘플링
                indices = np.random.choice(len(episode_data['states']), 
                                         min(batch_size, len(episode_data['states'])), 
                                         replace=False)
                
                states = [episode_data['states'][i] for i in indices]
                actions = [episode_data['actions'][i] for i in indices]
                rewards = [episode_data['rewards'][i] for i in indices]
                next_states = [episode_data['next_states'][i] for i in indices]
                dones = [episode_data['dones'][i] for i in indices]
                
                # 훈련 스텝
                loss_info = self.agent.train_step(states, actions, rewards, next_states, dones)
                
                # 히스토리 저장
                training_history['episode_rewards'].append(np.mean(rewards))
                training_history['actor_losses'].append(loss_info['actor_loss'])
                training_history['critic_losses'].append(loss_info['critic_loss'])
                training_history['entropies'].append(loss_info['entropy'])
                
                if episode % 100 == 0:
                    print(f"Episode {episode}: Reward={np.mean(rewards):.3f}, "
                          f"Actor Loss={loss_info['actor_loss']:.3f}, "
                          f"Critic Loss={loss_info['critic_loss']:.3f}")
        
        # 훈련 결과 시각화
        self.visualize_training_results(training_history)
        
        return training_history
    
    def visualize_training_results(self, history):
        """훈련 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 에피소드 보상
        axes[0, 0].plot(history['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Actor 손실
        axes[0, 1].plot(history['actor_losses'])
        axes[0, 1].set_title('Actor Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        
        # Critic 손실
        axes[1, 0].plot(history['critic_losses'])
        axes[1, 0].set_title('Critic Loss')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        
        # 엔트로피
        axes[1, 1].plot(history['entropies'])
        axes[1, 1].set_title('Entropy')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Entropy')
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.close()
    
    def save_model(self, path):
        """모델 저장"""
        torch.save({
            'agent_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'memory_state': self.agent.memory.get_memory_state()
        }, path)
        print(f"모델이 {path}에 저장되었습니다.")
    
    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path)
        self.agent.load_state_dict(checkpoint['agent_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"모델이 {path}에서 로드되었습니다.")

def main():
    """메인 실행 함수"""
    # 에이전트 초기화
    state_dim = 150  # 상태 벡터 차원 (조정 필요)
    action_dim = 4   # 4방향 이동
    agent = ElderlyRLAgent(state_dim, action_dim)
    
    # 훈련기 초기화
    data_dir = "beameyetracker/kiosk_data"
    trainer = ElderlyRLTrainer(agent, data_dir)
    
    # 훈련 데이터 로드
    trainer.load_training_data()
    
    # 강화학습 훈련
    training_history = trainer.train(num_episodes=500)
    
    # 모델 저장
    trainer.save_model('elderly_rl_agent.pth')
    
    print("강화학습 훈련 완료!")

if __name__ == "__main__":
    main() 