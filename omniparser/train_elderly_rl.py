#!/usr/bin/env python3
"""
고령자 강화학습 에이전트 훈련 스크립트

이 스크립트는 수집된 UIUX 인증 데이터를 사용하여
고령자 특성을 반영한 강화학습 에이전트를 훈련합니다.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from reinforcement_learning_agent import ElderlyRLAgent, ElderlyRLTrainer

def load_training_data_from_directory(data_dir):
    """데이터 디렉토리에서 훈련 데이터를 로드합니다."""
    print(f"데이터 디렉토리에서 훈련 데이터 로드 중: {data_dir}")
    
    training_data = []
    
    # 데이터 디렉토리 탐색
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            if dir_name.startswith('uiux_certification_'):
                data_path = os.path.join(root, dir_name)
                print(f"데이터 경로 발견: {data_path}")
                
                # 필요한 파일들 확인
                required_files = ['scenario_data.json', 'gaze_data.json', 'click_events.json']
                missing_files = []
                
                for file_name in required_files:
                    file_path = os.path.join(data_path, file_name)
                    if not os.path.exists(file_path):
                        missing_files.append(file_name)
                
                if missing_files:
                    print(f"  ⚠️ 누락된 파일: {missing_files}")
                    continue
                
                # 데이터 로드
                try:
                    # 시나리오 데이터 로드
                    scenario_file = os.path.join(data_path, 'scenario_data.json')
                    with open(scenario_file, 'r', encoding='utf-8') as f:
                        scenario_data = json.load(f)
                    
                    # 시선 데이터 로드
                    gaze_file = os.path.join(data_path, 'gaze_data.json')
                    with open(gaze_file, 'r', encoding='utf-8') as f:
                        gaze_data = json.load(f)
                    
                    # 클릭 이벤트 로드
                    click_file = os.path.join(data_path, 'click_events.json')
                    with open(click_file, 'r', encoding='utf-8') as f:
                        click_events = json.load(f)
                    
                    # 타임라인 데이터 로드 (선택사항)
                    timeline_file = os.path.join(data_path, 'timeline_data.json')
                    timeline_data = None
                    if os.path.exists(timeline_file):
                        with open(timeline_file, 'r', encoding='utf-8') as f:
                            timeline_data = json.load(f)
                    
                    # 성능 지표 로드 (선택사항)
                    metrics_file = os.path.join(data_path, 'performance_metrics.json')
                    performance_metrics = None
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r', encoding='utf-8') as f:
                            performance_metrics = json.load(f)
                    
                    # 데이터 유효성 검사
                    if len(gaze_data) < 10:  # 최소 10개의 시선 데이터 포인트 필요
                        print(f"  ⚠️ 시선 데이터가 너무 적습니다: {len(gaze_data)}개")
                        continue
                    
                    # 훈련 데이터로 변환
                    episode_data = convert_to_training_data(
                        scenario_data, gaze_data, click_events, timeline_data, performance_metrics
                    )
                    
                    if episode_data and len(episode_data['states']) > 0:
                        training_data.append(episode_data)
                        print(f"  ✅ 데이터 로드 완료: {len(episode_data['states'])}개 상태")
                    else:
                        print(f"  ⚠️ 유효한 훈련 데이터가 없습니다")
                
                except Exception as e:
                    print(f"  ❌ 데이터 로드 실패: {e}")
                    continue
    
    print(f"총 {len(training_data)}개의 에피소드 데이터 로드 완료")
    return training_data

def convert_to_training_data(scenario_data, gaze_data, click_events, timeline_data=None, performance_metrics=None):
    """실제 데이터를 강화학습 훈련 데이터로 변환합니다."""
    episode = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'dones': [],
        'metadata': {
            'scenario_steps': scenario_data.get('scenario_steps', []),
            'completed_steps': scenario_data.get('scenario_states', {}).get('completed_steps', []),
            'failed_steps': scenario_data.get('scenario_states', {}).get('failed_steps', []),
            'total_gaze_points': len(gaze_data),
            'total_clicks': len(click_events),
            'session_duration': scenario_data.get('session_duration', 0)
        }
    }
    
    # 시선 데이터를 기반으로 상태 시퀀스 생성
    for i, gaze_point in enumerate(gaze_data):
        # 상태 생성
        state = create_state_from_gaze(gaze_point, scenario_data, click_events)
        
        # 행동 생성 (시선 이동 방향)
        action = create_action_from_gaze(gaze_point, gaze_data, i)
        
        # 보상 생성 (고령자 특성 반영)
        reward = create_elderly_reward(gaze_point, click_events, scenario_data, i)
        
        # 다음 상태 (간단히 같은 상태로 설정)
        next_state = state
        
        # 종료 조건
        done = i == len(gaze_data) - 1
        
        episode['states'].append(state)
        episode['actions'].append(action)
        episode['rewards'].append(reward)
        episode['next_states'].append(next_state)
        episode['dones'].append(done)
    
    return episode

def create_state_from_gaze(gaze_point, scenario_data, click_events):
    """시선 데이터로부터 상태 벡터를 생성합니다."""
    # 기본 상태 벡터 (150차원)
    state_vector = []
    
    # 1. 시선 위치 (2차원)
    x_norm = gaze_point['x'] / 1920 if 'x' in gaze_point else 0.5
    y_norm = gaze_point['y'] / 1080 if 'y' in gaze_point else 0.5
    state_vector.extend([x_norm, y_norm])
    
    # 2. 시선 신뢰도 (1차원)
    confidence = gaze_point.get('confidence', 0.5)
    state_vector.append(confidence)
    
    # 3. 시나리오 진행률 (1차원)
    current_step = scenario_data.get('scenario_states', {}).get('current_step', 0)
    total_steps = len(scenario_data.get('scenario_steps', []))
    progress = current_step / max(total_steps, 1)
    state_vector.append(progress)
    
    # 4. 시선 속도 (1차원)
    velocity = gaze_point.get('velocity', 0.0) / 1000.0  # 정규화
    state_vector.append(velocity)
    
    # 5. 시선 가속도 (1차원)
    acceleration = gaze_point.get('acceleration', 0.0) / 10000.0  # 정규화
    state_vector.append(acceleration)
    
    # 6. 최근 클릭과의 시간 차이 (1차원)
    time_diff = 0.0
    if click_events:
        current_time = gaze_point.get('timestamp', 0)
        for click_event in click_events:
            click_time = click_event.get('timestamp', 0)
            diff = abs(current_time - click_time)
            if diff < time_diff or time_diff == 0:
                time_diff = diff
    time_diff_norm = min(time_diff / 10.0, 1.0)  # 최대 10초로 정규화
    state_vector.append(time_diff_norm)
    
    # 7. 시선 고정 시간 (1차원)
    fixation_duration = gaze_point.get('fixation_duration', 0.0) / 5.0  # 최대 5초로 정규화
    state_vector.append(fixation_duration)
    
    # 8. 시나리오 단계별 정보 (최대 10단계)
    scenario_steps = scenario_data.get('scenario_steps', [])
    for i in range(10):
        if i < len(scenario_steps):
            # 단계 완료 여부 (1차원)
            completed = 1.0 if i in scenario_data.get('scenario_states', {}).get('completed_steps', []) else 0.0
            state_vector.append(completed)
        else:
            state_vector.append(0.0)
    
    # 9. 메모리 상태 (간단한 시뮬레이션)
    memory_items = min(len(scenario_steps), 5)  # 최대 5개 아이템
    memory_confidence = 0.8 if memory_items > 0 else 0.0
    state_vector.extend([memory_items / 5.0, memory_confidence])
    
    # 10. 나머지를 0으로 패딩하여 150차원 맞추기
    while len(state_vector) < 150:
        state_vector.append(0.0)
    
    return torch.FloatTensor(state_vector)

def create_action_from_gaze(gaze_point, gaze_data, index):
    """시선 이동으로부터 행동을 생성합니다."""
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

def create_elderly_reward(gaze_point, click_events, scenario_data, step_index):
    """고령자 특성을 반영한 보상을 생성합니다."""
    reward = 0.0
    
    # 1. 클릭 이벤트와의 시간 차이 보상
    for click_event in click_events:
        time_diff = abs(gaze_point['timestamp'] - click_event['timestamp'])
        if time_diff < 1.0:  # 1초 이내
            reward += 5.0  # 클릭 전후 시선 보상
        elif time_diff < 3.0:  # 3초 이내
            reward += 2.0  # 중간 보상
    
    # 2. 시선 고정 시간 보상 (고령자는 천천히 보는 것이 자연스러움)
    if 'velocity' in gaze_point and gaze_point['velocity'] < 100:
        reward += 1.0  # 천천히 움직이는 것에 보상
    elif 'velocity' in gaze_point and gaze_point['velocity'] > 1000:
        reward -= 1.0  # 너무 빠르게 움직이는 것에 패널티
    
    # 3. 시선 신뢰도 보상
    if 'confidence' in gaze_point:
        if gaze_point['confidence'] > 0.8:
            reward += 2.0
        elif gaze_point['confidence'] < 0.3:
            reward -= 1.0
    
    # 4. 시나리오 진행 보상
    scenario_states = scenario_data.get('scenario_states', {})
    completed_steps = scenario_states.get('completed_steps', [])
    if step_index > 0 and len(completed_steps) > 0:
        reward += len(completed_steps) * 0.5  # 완료된 단계당 보상
    
    # 5. 반복 탐색 패널티 (같은 위치에 오래 머무름)
    if step_index > 10:
        recent_positions = []
        for i in range(max(0, step_index - 10), step_index):
            if i < len(gaze_data):
                pos = (gaze_data[i]['x'], gaze_data[i]['y'])
                recent_positions.append(pos)
        
        # 같은 위치에 오래 머무르면 패널티
        if len(set(recent_positions)) < 3:  # 3개 미만의 고유 위치
            reward -= 2.0
    
    # 6. 목표 달성 보상
    current_step = scenario_states.get('current_step', 0)
    total_steps = len(scenario_data.get('scenario_steps', []))
    if current_step >= total_steps:
        reward += 10.0  # 모든 단계 완료 보상
    
    return reward

def train_elderly_rl_agent(training_data, num_episodes=1000, batch_size=32, learning_rate=0.001):
    """고령자 강화학습 에이전트를 훈련합니다."""
    print(f"강화학습 훈련 시작...")
    print(f"- 에피소드 수: {num_episodes}")
    print(f"- 배치 크기: {batch_size}")
    print(f"- 학습률: {learning_rate}")
    
    # 에이전트 초기화
    state_dim = 150
    action_dim = 4
    agent = ElderlyRLAgent(state_dim, action_dim)
    agent.learning_rate = learning_rate
    
    # 훈련기 초기화
    trainer = ElderlyRLTrainer(agent, None)  # data_dir는 None으로 설정
    trainer.training_data = training_data  # 직접 데이터 설정
    
    # 훈련 실행
    training_history = trainer.train(num_episodes=num_episodes, batch_size=batch_size)
    
    return agent, training_history

def visualize_training_progress(history, save_path='training_progress.png'):
    """훈련 진행 상황을 시각화합니다."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 에피소드 보상
    axes[0, 0].plot(history['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Actor 손실
    axes[0, 1].plot(history['actor_losses'])
    axes[0, 1].set_title('Actor Loss')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Critic 손실
    axes[1, 0].plot(history['critic_losses'])
    axes[1, 0].set_title('Critic Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # 엔트로피
    axes[1, 1].plot(history['entropies'])
    axes[1, 1].set_title('Entropy')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"훈련 진행 상황이 {save_path}에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='고령자 강화학습 에이전트 훈련')
    parser.add_argument('--data_dir', type=str, default='beameyetracker/kiosk_data',
                       help='훈련 데이터 디렉토리 경로')
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='훈련 에피소드 수')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='학습률')
    parser.add_argument('--output_dir', type=str, default='trained_models',
                       help='훈련된 모델 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 훈련 데이터 로드
    training_data = load_training_data_from_directory(args.data_dir)
    
    if not training_data:
        print("❌ 훈련 데이터를 찾을 수 없습니다.")
        print(f"데이터 디렉토리를 확인해주세요: {args.data_dir}")
        return
    
    print(f"✅ {len(training_data)}개의 에피소드 데이터 로드 완료")
    
    # 데이터 통계 출력
    total_states = sum(len(episode['states']) for episode in training_data)
    total_actions = sum(len(episode['actions']) for episode in training_data)
    avg_reward = np.mean([np.mean(episode['rewards']) for episode in training_data])
    
    print(f"데이터 통계:")
    print(f"- 총 상태 수: {total_states}")
    print(f"- 총 행동 수: {total_actions}")
    print(f"- 평균 보상: {avg_reward:.3f}")
    
    # 강화학습 훈련
    print("\n🚀 강화학습 훈련 시작...")
    agent, training_history = train_elderly_rl_agent(
        training_data,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # 훈련 결과 시각화
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_path = os.path.join(args.output_dir, f'training_progress_{timestamp}.png')
    visualize_training_progress(training_history, progress_path)
    
    # 모델 저장
    model_path = os.path.join(args.output_dir, f'elderly_rl_agent_{timestamp}.pth')
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'memory_state': agent.memory.get_memory_state(),
        'training_history': training_history,
        'training_config': {
            'num_episodes': args.num_episodes,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'data_dir': args.data_dir
        }
    }, model_path)
    
    print(f"\n✅ 훈련 완료!")
    print(f"- 모델 저장 경로: {model_path}")
    print(f"- 훈련 진행 상황: {progress_path}")
    
    # 최종 통계
    final_rewards = training_history['episode_rewards'][-100:]  # 마지막 100 에피소드
    print(f"\n최종 통계:")
    print(f"- 마지막 100 에피소드 평균 보상: {np.mean(final_rewards):.3f}")
    print(f"- 최고 보상: {np.max(training_history['episode_rewards']):.3f}")
    print(f"- 최저 보상: {np.min(training_history['episode_rewards']):.3f}")

if __name__ == "__main__":
    main() 