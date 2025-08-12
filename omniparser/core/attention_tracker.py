import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import json
import os
from datetime import datetime

class AttentionTracker:
    def __init__(self, grid_size: Tuple[int, int] = (10, 10)):
        """
        Initialize the attention tracker with a grid-based vision system.
        
        Args:
            grid_size: Tuple of (height, width) for the vision grid
        """
        self.grid_size = grid_size
        self.vision_grid = np.zeros(grid_size)
        self.attention_history = []
        self.stamp_position_history = []
        self.button_positions = {}
        
    def update_vision_grid(self, stamp_position, visible_buttons):
        """Update vision grid based on current stamp position and visible buttons"""
        # Update stamp position history
        self.stamp_position_history.append(stamp_position)
        
        # Calculate attention scores for each button
        for button in visible_buttons:
            # Calculate button center from bbox coordinates
            bbox = button['bbox']
            button_center = (
                (bbox[0] + bbox[2]) / 2,  # x center
                (bbox[1] + bbox[3]) / 2   # y center
            )
            
            # Calculate attention score
            attention_score = self._calculate_attention_score(stamp_position, button_center)
            
            # Update attention history
            self.attention_history.append({
                'timestamp': datetime.now().isoformat(),
                'stamp_position': stamp_position,
                'button_center': button_center,
                'attention_score': attention_score
            })
            
            # Update vision grid: (height, width) 튜플의 각 요소를 분리하여 곱함
            grid_x = int(button_center[0] * self.grid_size[1])  # width 방향
            grid_y = int(button_center[1] * self.grid_size[0])  # height 방향
            
            if 0 <= grid_x < self.grid_size[1] and 0 <= grid_y < self.grid_size[0]:
                self.vision_grid[grid_y, grid_x] = max(
                    self.vision_grid[grid_y, grid_x],
                    attention_score
                )
        
        return self.vision_grid.copy()
    
    def _calculate_attention_score(self, stamp_pos: Tuple[float, float], 
                                 button_center: Tuple[float, float]) -> float:
        """
        Calculate attention score based on distance between stamp and button.
        
        Args:
            stamp_pos: Current stamp position
            button_center: Button center position
            
        Returns:
            Attention score (higher for closer objects)
        """
        distance = np.sqrt((stamp_pos[0] - button_center[0])**2 + 
                         (stamp_pos[1] - button_center[1])**2)
        return 1.0 / (distance + 1e-5)
    
    def visualize_attention(self, save_path: str = None) -> None:
        """
        Visualize the attention grid and stamp movement history.
        
        Args:
            save_path: Optional path to save the visualization
        """
        plt.figure(figsize=(10, 8))
        
        # Plot attention grid
        plt.imshow(self.vision_grid, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Attention Score')
        
        # Plot stamp movement history
        if self.stamp_position_history:
            history = np.array(self.stamp_position_history)
            plt.plot(history[:, 0] * self.grid_size[1], 
                    history[:, 1] * self.grid_size[0], 
                    'b-', alpha=0.5, label='Stamp Movement')
            plt.plot(history[-1, 0] * self.grid_size[1], 
                    history[-1, 1] * self.grid_size[0], 
                    'bo', label='Current Position')
        
        plt.title('Attention Grid and Stamp Movement')
        plt.xlabel('Grid X')
        plt.ylabel('Grid Y')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def save_state(self, output_dir: str) -> None:
        """
        Save the current state of the attention tracker.
        
        Args:
            output_dir: Directory to save the state
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state = {
            'grid_size': self.grid_size,
            'vision_grid': self.vision_grid.tolist(),
            'stamp_position_history': self.stamp_position_history,
            'attention_history': [grid.tolist() for grid in self.attention_history]
        }
        
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'attention_state_{timestamp}.json'), 'w') as f:
            json.dump(state, f)

    def update_stamp_position(self, position: Tuple[int, int]):
        """도장 위치 업데이트"""
        # Update stamp position history
        self.stamp_position_history.append(position)
        
        # Update vision grid based on new position
        if self.button_positions:
            self.update_vision_grid(position, self.button_positions)
            
    def update_button_positions(self, buttons: List[Dict]):
        """버튼 위치 정보 업데이트"""
        self.button_positions = buttons
        if self.stamp_position_history:
            self.update_vision_grid(self.stamp_position_history[-1], buttons)
            
    def calculate_button_attention(self, button_center: Tuple[int, int]) -> float:
        """버튼에 대한 attention score 계산"""
        if not self.stamp_position_history:
            return 0.0
            
        return self._calculate_attention_score(self.stamp_position_history[-1], button_center) 