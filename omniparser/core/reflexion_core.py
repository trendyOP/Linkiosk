import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime

from .attention_tracker import AttentionTracker
from .action_logger import ActionLogger
from .explanation_generator import ExplanationGenerator

class ReflexionCore:
    def __init__(self,
                 grid_size: Tuple[int, int] = (10, 10),
                 log_dir: str = "logs",
                 template_dir: str = "templates"):
        """
        Initialize the reflexion core that integrates all components.
        
        Args:
            grid_size: Size of the vision grid
            log_dir: Directory for logging
            template_dir: Directory for explanation templates
        """
        self.grid_size = grid_size
        self.log_dir = log_dir
        self.template_dir = template_dir
        
        # 컴포넌트 초기화
        self.attention_tracker = AttentionTracker(grid_size=grid_size)
        self.action_logger = ActionLogger(log_dir=log_dir)
        self.explanation_generator = ExplanationGenerator(template_dir=template_dir)
        
        # 🆕 스텝 단위 로그를 저장할 리스트 초기화
        self.step_logs: List[Dict] = []
        # 🆕 보상 누적 기록 (optional)
        self.reward_history: List[float] = []
        
        # 실패 포인트 기록
        self.failure_points = []
        
        # 로그 디렉토리 생성
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(template_dir, exist_ok=True)
        
        self.current_episode = []
        
    def process_step(self,
                    stamp_position: Tuple[float, float],
                    detected_buttons: List[Dict[str, Any]],
                    action: int,
                    reward: float,
                    q_values: Optional[np.ndarray] = None,
                    policy_probs: Optional[np.ndarray] = None,
                    ocr_text: Optional[List[str]] = None,
                    ocr_bbox: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        한 스텝 처리 후 로그 및 리플렉션용 데이터를 준비합니다.
        
        Args:
            stamp_position: 현재 스탬프 위치 (x, y)
            detected_buttons: 감지된 버튼 목록
            action: 선택된 액션
            reward: 받은 보상
            q_values: Q-값 (선택사항)
            policy_probs: 정책 확률 (선택사항)
            ocr_text: OCR로 감지된 텍스트 목록 (선택사항)
            ocr_bbox: OCR 바운딩 박스 목록 (선택사항)
            
        Returns:
            Dict[str, Any]: 스텝 처리 결과 정보
        """
        # 스텝 데이터 준비
        step_data = {
            'timestamp': datetime.now().isoformat(),
            'stamp_position': stamp_position,
            'detected_buttons': detected_buttons,
            'action': action,
            'reward': reward
        }
        
        # OCR 정보가 있으면 추가
        if ocr_text is not None and ocr_bbox is not None:
            step_data['ocr_results'] = {
                'texts': ocr_text,
                'boxes': ocr_bbox
            }
        
        # Q-값과 정책 확률이 있으면 추가
        if q_values is not None:
            step_data['q_values'] = q_values.tolist()
        if policy_probs is not None:
            step_data['policy_probs'] = policy_probs.tolist()
        
        # 스텝 로그에 추가
        self.step_logs.append(step_data)
        
        # 실패 분석
        failure_explanation = None
        if reward < 0:
            failure_explanation = self._generate_failure_explanation(action, detected_buttons)
            if failure_explanation:
                step_data['failure_explanation'] = failure_explanation
        
        # 개선 제안
        suggestion = self._generate_suggestion(action)
        if suggestion:
            step_data['suggestion'] = suggestion
        
        return step_data
        
    def end_episode(self, total_reward: float) -> Dict[str, Any]:
        """
        End the current episode and generate reflections.
        
        Args:
            total_reward: Total reward for the episode
            
        Returns:
            Dictionary containing episode summary and reflections
        """
        # Log episode end
        self.action_logger.end_episode(total_reward)
        
        # Generate reflection
        episode_data = {
            'steps': self.current_episode,
            'total_reward': total_reward
        }
        reflection = self.explanation_generator.generate_reflection(
            episode_data, self.failure_points)
            
        # Save reflection
        reflection_path = self.explanation_generator.save_explanation(
            reflection, os.path.join(self.action_logger.log_dir, 'reflections'))
            
        # Generate visualizations
        self.attention_tracker.visualize_attention(
            os.path.join(self.action_logger.log_dir, 'attention_heatmap.png'))
        self.action_logger.visualize_rewards(
            os.path.join(self.action_logger.log_dir, 'reward_history.png'))
        self.action_logger.visualize_action_distribution(
            os.path.join(self.action_logger.log_dir, 'action_distribution.png'))
            
        # Reset episode data
        self.current_episode = []
        self.failure_points = []
        
        return {
            'reflection': reflection,
            'reflection_path': reflection_path,
            'statistics': self.action_logger.get_statistics()
        }
        
    def _generate_failure_explanation(self,
                                    action: int,
                                    detected_buttons: List[Dict[str, Any]]) -> str:
        """
        Generate explanation for a failure.
        
        Args:
            action: Failed action
            detected_buttons: List of detected buttons
            
        Returns:
            Explanation string
        """
        if not detected_buttons:
            return self.explanation_generator.generate_explanation(
                'button_not_found',
                {'button_name': 'unknown'}
            )
            
        # Check if click failed
        if action == 'click':
            return self.explanation_generator.generate_explanation(
                'click_failure',
                {
                    'button_name': 'unknown',
                    'failure_reason': '클릭 위치가 버튼 중심에서 벗어남',
                    'vision_state': '시야 내 버튼 존재'
                }
            )
            
        return self.explanation_generator.generate_explanation(
            'general_failure',
            {
                'failure_reason': '알 수 없는 실패',
                'current_state': f'행동: {action}'
            }
        )
        
    def _generate_suggestion(self,
                           action: int) -> str:
        """
        Generate suggestion for improvement.
        
        Args:
            action: Failed action
            
        Returns:
            Suggestion string
        """
        if action == 'click':
            return "다음에는 버튼 중심에 더 가깝게 클릭하세요."
        elif action == 'move':
            return "다음에는 더 천천히, 정확한 위치로 이동하세요."
        else:
            return "다음에는 더 신중하게 행동을 선택하세요." 