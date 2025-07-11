import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import threading
from collections import deque
from PIL import ImageGrab, Image
import io
import base64

try:
    from eyeware import beam_eye_tracker as bet
    from screeninfo import get_monitors
    EYEWARE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Eyeware SDK not available. Eye tracking will be disabled. Error: {e}")
    EYEWARE_AVAILABLE = False

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError as e:
    print(f"Warning: keyboard module not available. Install with: pip install keyboard. Error: {e}")
    KEYBOARD_AVAILABLE = False

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ClickSegmentCollector:
    def __init__(self, scenario_name="click_segment_scenario"):
        self.scenario_name = scenario_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = f"kiosk_data/{scenario_name}_{self.timestamp}"
        os.makedirs(self.data_dir, exist_ok=True)
        self.gaze_data = []
        self.click_events = []
        self.screenshot_paths = []
        self.tracker = None
        self.tracking_active = False
        self.screen_width, self.screen_height = self.get_screen_resolution()
        self.log(f"Initialized collector: {self.data_dir}")

    def log(self, msg):
        print(f"[LOG] {msg}")

    def get_screen_resolution(self):
        if EYEWARE_AVAILABLE:
            primary_monitor = next((m for m in get_monitors() if m.is_primary), get_monitors()[0])
            return primary_monitor.width, primary_monitor.height
        return 1920, 1080

    def start_eye_tracker(self):
        if not EYEWARE_AVAILABLE:
            self.log("Eyeware SDK not available.")
            return
        viewport = bet.ViewportGeometry()
        viewport.point_00.x = 0
        viewport.point_00.y = 0
        viewport.point_11.x = self.screen_width
        viewport.point_11.y = self.screen_height
        self.tracker = bet.API("click-segment-app", viewport)
        self.tracker.attempt_starting_the_beam_eye_tracker()
        self.tracking_active = True
        self.log("Started eye tracker.")

    def start_gaze_thread(self):
        if not self.tracker:
            return
        self.gaze_thread = threading.Thread(target=self._gaze_loop)
        self.gaze_thread.daemon = True
        self.gaze_thread.start()

    def _gaze_loop(self):
        last_update_timestamp = bet.NULL_DATA_TIMESTAMP()
        while self.tracking_active:
            if self.tracker.wait_for_new_tracking_state_set(last_update_timestamp, 100):
                tracking_state_set = self.tracker.get_latest_tracking_state_set()
                user_state = tracking_state_set.user_state()
                if (user_state.unified_screen_gaze.confidence != bet.TrackingConfidence.LOST_TRACKING and
                    user_state.timestamp_in_seconds != bet.NULL_DATA_TIMESTAMP()):
                    gaze = user_state.unified_screen_gaze.point_of_regard
                    gaze_data_point = {
                        'timestamp': time.time(),
                        'x': gaze.x,
                        'y': gaze.y,
                        'confidence': user_state.unified_screen_gaze.confidence
                    }
                    self.gaze_data.append(gaze_data_point)
                last_update_timestamp = user_state.timestamp_in_seconds
            time.sleep(0.01)

    def record_click(self, x, y, element_text=""):
        ts = time.time()
        screenshot_path = os.path.join(self.data_dir, f"click_{len(self.click_events)+1}_screenshot.png")
        img = ImageGrab.grab()
        img.save(screenshot_path)
        click_event = {
            'timestamp': ts,
            'x': x,
            'y': y,
            'element_text': element_text,
            'screenshot_path': screenshot_path
        }
        self.click_events.append(click_event)
        self.screenshot_paths.append(screenshot_path)
        self.log(f"Click recorded at ({x},{y}) with screenshot: {screenshot_path}")

    def save_click_gaze_segments(self):
        # 클릭 구간별 시선 데이터 분할 및 저장
        if not self.click_events:
            self.log("No click events to segment.")
            return
        segments = []
        for i, click in enumerate(self.click_events):
            t_start = click['timestamp']
            t_end = self.click_events[i+1]['timestamp'] if i+1 < len(self.click_events) else float('inf')
            gaze_segment = [g for g in self.gaze_data if t_start <= g['timestamp'] < t_end]
            seg_path = os.path.join(self.data_dir, f"click_{i+1}_gaze.json")
            with open(seg_path, 'w', encoding='utf-8') as f:
                json.dump(gaze_segment, f, ensure_ascii=False, indent=2)
            segments.append({'click': click, 'gaze': gaze_segment, 'gaze_path': seg_path})
            self.log(f"Saved gaze segment for click {i+1} ({len(gaze_segment)} points)")
        return segments

    def visualize_click_segments(self, segments):
        for i, seg in enumerate(segments):
            screenshot_path = seg['click']['screenshot_path']
            gaze_points = seg['gaze']
            if not os.path.exists(screenshot_path):
                continue
            img = Image.open(screenshot_path)
            img_array = np.array(img)
            plt.figure(figsize=(12, 8))
            plt.imshow(img_array)
            if gaze_points:
                x_coords = [g['x'] for g in gaze_points]
                y_coords = [g['y'] for g in gaze_points]
                h = plt.hist2d(x_coords, y_coords, bins=30, cmap='hot', alpha=0.6)
                plt.colorbar(h[3], label='시선 빈도')
            plt.scatter([seg['click']['x']], [seg['click']['y']], c='cyan', s=200, marker='*', label='클릭 위치', zorder=5)
            plt.title(f'클릭 {i+1} 구간 시각화')
            plt.xlabel('X 좌표 (픽셀)')
            plt.ylabel('Y 좌표 (픽셀)')
            plt.legend()
            plt.tight_layout()
            out_path = os.path.join(self.data_dir, f"click_{i+1}_heatmap.png")
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.log(f"Saved heatmap: {out_path}")

    def save_all(self):
        # 전체 시선/클릭 데이터 저장
        with open(os.path.join(self.data_dir, "gaze_data.json"), 'w', encoding='utf-8') as f:
            json.dump(self.gaze_data, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.data_dir, "click_events.json"), 'w', encoding='utf-8') as f:
            json.dump(self.click_events, f, ensure_ascii=False, indent=2)
        self.log("Saved all gaze and click data.")

    def run(self):
        self.start_eye_tracker()
        self.start_gaze_thread()
        self.log("Press 'c' to record a click at mouse position, ESC to finish.")
        try:
            while True:
                if keyboard.is_pressed('esc'):
                    break
                if keyboard.is_pressed('c'):
                    # 마우스 위치 가져오기 (pyautogui 필요)
                    try:
                        import pyautogui
                        x, y = pyautogui.position()
                        self.record_click(x, y)
                        time.sleep(0.5)  # 중복 방지
                    except ImportError:
                        self.log("pyautogui가 필요합니다: pip install pyautogui")
                        break
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        self.tracking_active = False
        if hasattr(self, 'gaze_thread'):
            self.gaze_thread.join(timeout=1.0)
        self.save_all()
        segments = self.save_click_gaze_segments()
        if segments:
            self.visualize_click_segments(segments)
        self.log("모든 작업 완료!")

if __name__ == "__main__":
    collector = ClickSegmentCollector("click_segment_scenario")
    collector.run()