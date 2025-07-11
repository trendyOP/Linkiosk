# Copyright (C) 2025 Eyeware Tech SA.
#
# All rights reserved.
#
# This file implements a wrapper of the C++ Beam Eye Tracker API.
# For the moment, please refer to the C++ headers for documentation.

from enum import IntEnum
from typing import overload
from numpy.typing import NDArray

class Timestamp:
    value: float

def NULL_DATA_TIMESTAMP() -> Timestamp: ...

class TrackingDataReceptionStatus(IntEnum):
    NOT_RECEIVING_TRACKING_DATA: int
    RECEIVING_TRACKING_DATA: int
    ATTEMPTING_TRACKING_AUTO_START: int

class TrackingConfidence(IntEnum):
    LOST_TRACKING: int
    LOW: int
    MEDIUM: int
    HIGH: int

class Version:
    major: int
    minor: int
    patch: int
    build: int

class Point:
    x: int
    y: int

class PointF:
    x: float
    y: float

class ViewportGeometry:
    point_00: Point
    point_11: Point

class Vector3D:
    x: float
    y: float
    z: float

class UnifiedScreenGaze:
    confidence: TrackingConfidence
    point_of_regard: Point
    unbounded_point_of_regard: Point

class ViewportGaze:
    confidence: TrackingConfidence
    normalized_point_of_regard: PointF

class HeadPose:
    confidence: TrackingConfidence
    rotation_from_hcs_to_wcs: NDArray
    translation_from_hcs_to_wcs: Vector3D
    track_session_uid: int

class SimCameraTransform3D:
    roll_in_radians: float
    pitch_in_radians: float
    yaw_in_radians: float
    x_in_meters: float
    y_in_meters: float
    z_in_meters: float

class UserState:
    timestamp_in_seconds: Timestamp
    head_pose: HeadPose
    unified_screen_gaze: UnifiedScreenGaze
    viewport_gaze: ViewportGaze

class SimGameCameraState:
    timestamp_in_seconds: Timestamp
    eye_tracking_pose_component: SimCameraTransform3D
    head_tracking_pose_component: SimCameraTransform3D

class GameImmersiveHUDState:
    timestamp_in_seconds: Timestamp
    looking_at_viewport_top_left: float
    looking_at_viewport_top_middle: float
    looking_at_viewport_top_right: float
    looking_at_viewport_center_left: float
    looking_at_viewport_center_right: float
    looking_at_viewport_bottom_left: float
    looking_at_viewport_bottom_middle: float
    looking_at_viewport_bottom_right: float

class TrackingStateSet:
    def user_state(self) -> UserState: ...
    def sim_game_camera_state(self) -> SimGameCameraState: ...
    def game_immersive_hud_state(self) -> GameImmersiveHUDState: ...

class TrackingListener:
    def __init__(self) -> None: ...
    def on_tracking_data_reception_status_changed(self, status: TrackingDataReceptionStatus) -> None: ...
    def on_tracking_state_set_update(self, tracking_state_set: TrackingStateSet, timestamp: float) -> None: ...

class API:
    def __init__(self, friendly_name: str, initial_viewport_geometry: ViewportGeometry) -> None: ...
    def get_version(self) -> Version: ...
    def update_viewport_geometry(self, new_viewport_geometry: ViewportGeometry) -> None: ...
    def attempt_starting_the_beam_eye_tracker(self) -> None: ...
    def start_receiving_tracking_data_on_listener(self, listener: TrackingListener) -> int: ...
    def stop_receiving_tracking_data_on_listener(self, listener_handle: int) -> None: ...
    @overload
    def wait_for_new_tracking_state_set(self, last_update_timestamp: float) -> bool: ...
    @overload
    def wait_for_new_tracking_state_set(self, last_update_timestamp: float, timeout_ms: int) -> bool: ...
    def get_latest_tracking_state_set(self) -> TrackingStateSet: ...
    def get_tracking_data_reception_status(self) -> TrackingDataReceptionStatus: ...
    @staticmethod
    def compute_sim_game_camera_transform_parameters(
        state: SimGameCameraState, eye_tracking_weight: float = 1.0, head_tracking_weight: float = 1.0
    ) -> SimCameraTransform3D: ...
    def recenter_sim_game_camera_start(self) -> bool: ...
    def recenter_sim_game_camera_end(self) -> None: ...
