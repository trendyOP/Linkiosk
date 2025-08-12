from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime

class ExplanationGenerator:
    def __init__(self, template_dir: str = "templates"):
        """
        Initialize the explanation generator.
        
        Args:
            template_dir: Directory containing explanation templates
        """
        self.template_dir = template_dir
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """
        Load explanation templates from files.
        
        Returns:
            Dictionary of template strings
        """
        templates = {
            'button_not_found': """
            도장은 '{button_name}' 버튼을 인식하지 못했습니다.
            시야에는 있었지만, 고령자 페르소나는 좁은 시야로 인해 중심을 인식하지 못했고,
            클릭은 너무 일찍 수행되었습니다.
            """,
            
            'click_failure': """
            도장이 '{button_name}' 버튼을 클릭하려 했으나 실패했습니다.
            원인: {failure_reason}
            시야 상태: {vision_state}
            """,
            
            'successful_action': """
            도장이 '{button_name}' 버튼을 성공적으로 클릭했습니다.
            시야 상태: {vision_state}
            수행된 행동: {action_taken}
            """,
            
            'general_failure': """
            작업 수행 중 오류가 발생했습니다.
            원인: {failure_reason}
            현재 상태: {current_state}
            """
        }
        
        # Create template directory if it doesn't exist
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Save templates to files
        for template_name, template_content in templates.items():
            template_path = os.path.join(self.template_dir, f"{template_name}.txt")
            if not os.path.exists(template_path):
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(template_content.strip())
                    
        return templates
    
    def generate_explanation(self,
                           event_type: str,
                           context: Dict[str, Any]) -> str:
        """
        Generate a natural language explanation for an event.
        
        Args:
            event_type: Type of event (e.g., 'button_not_found', 'click_failure')
            context: Dictionary containing context information
            
        Returns:
            Generated explanation string
        """
        if event_type not in self.templates:
            return self.generate_explanation('general_failure', {
                'failure_reason': f"Unknown event type: {event_type}",
                'current_state': str(context)
            })
            
        template = self.templates[event_type]
        return template.format(**context)
    
    def generate_reflection(self,
                          episode_data: Dict[str, Any],
                          failure_points: List[Dict[str, Any]]) -> str:
        """
        Generate a reflection on an episode's performance.
        
        Args:
            episode_data: Dictionary containing episode information
            failure_points: List of dictionaries containing failure information
            
        Returns:
            Generated reflection string
        """
        reflection = ["[회고]"]
        
        # Add general episode statistics
        total_steps = len(episode_data.get('steps', []))
        total_reward = episode_data.get('total_reward', 0)
        reflection.append(f"- 총 {total_steps}단계 수행, 총 보상: {total_reward:.2f}")
        
        # Add failure analysis
        for failure in failure_points:
            reflection.append(f"- {failure['description']}")
            if 'suggestion' in failure:
                reflection.append(f"  → {failure['suggestion']}")
                
        # Add success analysis if any
        if total_reward > 0:
            reflection.append("\n[성공 요인]")
            reflection.append("- 버튼 인식 및 클릭 성공")
            reflection.append("- 효율적인 경로 탐색")
            
        return "\n".join(reflection)
    
    def save_explanation(self, explanation: str, output_dir: str) -> str:
        """
        설명을 파일로 저장합니다.
        
        Args:
            explanation: 저장할 설명
            output_dir: 저장할 디렉토리
            
        Returns:
            저장된 파일의 경로
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(output_dir, f'explanation_{timestamp}.txt')
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(explanation)
                
            return file_path
        except Exception as e:
            print(f"Warning: Failed to save explanation: {str(e)}")
            return None 