import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

class ActionLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.logs: List[Dict] = []
        self.q_values: Dict = {}
        self.policy_probs: Dict = {}
        self.reward_history: List[float] = []
        self.action_counts: Dict[str, int] = {}
        
        # 로그 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)
        
    def log_step(self, 
                 stamp_position: tuple,
                 vision_state: dict,
                 detected_buttons: list,
                 selected_action: str,
                 reward: float,
                 q_values: dict = None,
                 policy_probs: dict = None):
        """각 step의 정보를 로그에 저장"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'stamp_position': stamp_position,
            'vision_state': vision_state,
            'detected_buttons': detected_buttons,
            'selected_action': selected_action,
            'reward': reward
        }
        
        if q_values:
            self.q_values = q_values
            log_entry['q_values'] = q_values
            
        if policy_probs:
            self.policy_probs = policy_probs
            log_entry['policy_probs'] = policy_probs
            
        self.logs.append(log_entry)
        self.reward_history.append(reward)
        self.action_counts[selected_action] = self.action_counts.get(selected_action, 0) + 1
        self._save_logs()
        
    def end_episode(self, total_reward: float):
        """에피소드 종료 시 통계 저장"""
        stats = {
            'total_reward': total_reward,
            'average_reward': np.mean(self.reward_history),
            'action_distribution': self.action_counts,
            'episode_length': len(self.logs)
        }
        
        # 통계 저장
        stats_file = os.path.join(self.log_dir, f'stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
            
        # 로그 초기화
        self.logs = []
        self.reward_history = []
        self.action_counts = {}
        
    def get_recent_logs(self, n: int = 5) -> List[Dict]:
        """최근 n개의 로그 반환"""
        return self.logs[-n:] if self.logs else []
        
    def get_statistics(self) -> Dict:
        """현재까지의 통계 반환"""
        return {
            'total_actions': sum(self.action_counts.values()),
            'action_distribution': self.action_counts,
            'average_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'total_reward': sum(self.reward_history)
        }
        
    def _save_logs(self):
        """로그를 파일에 저장"""
        log_file = os.path.join(self.log_dir, 'action_log.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)
            
    def visualize_q_values(self, save_path: str = None):
        """Q-value 시각화"""
        if not self.q_values:
            return
            
        actions = list(self.q_values.keys())
        values = [float(self.q_values[action]) for action in actions]
        
        plt.figure(figsize=(10, 6))
        plt.bar(actions, values)
        plt.title('Q-values for Actions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def visualize_policy_probs(self, save_path: str = None):
        """Policy 확률 시각화"""
        if not self.policy_probs:
            return
            
        actions = list(self.policy_probs.keys())
        probs = [float(self.policy_probs[action]) for action in actions]
        
        plt.figure(figsize=(10, 6))
        plt.bar(actions, probs)
        plt.title('Policy Probabilities for Actions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def visualize_rewards(self, save_path: str = None):
        """보상 히스토리 시각화"""
        if not self.reward_history:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.reward_history)
        plt.title('Reward History')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def visualize_action_distribution(self, save_path: str = None):
        """액션 분포 시각화"""
        if not self.action_counts:
            return
            
        actions = list(self.action_counts.keys())
        counts = list(self.action_counts.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(actions, counts)
        plt.title('Action Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close() 