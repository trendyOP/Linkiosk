# Copyright (c) 2025, Eyeware Tech SA.
#
# All rights reserved.
#
# This sample demonstrates how to use the asynchronous data access method.
#
# Asynchronous data access is convenient in case you want to get a TrackingStateSet at low
# latency, and you do not want your main thread waiting for data updates.
#
# It is specially convenient when you already have a class holding to manage the API and you simply
# inherit TrackingListener and override the virtual methods.
#
# WARNING: the TrackingListener instance must be alive in between the
# start_receiving_tracking_data_on_listener and the stop_receiving_tracking_data_on_listener calls.
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import bet_add_to_python_path

import time
from eyeware import beam_eye_tracker
from bet_sample_utils import (
    print_tracking_data_reception_status,
    print_latest_tracking_state_set,
    print_tracking_data_reception_status_if_changed,
)


class MyTrackingListener(beam_eye_tracker.TrackingListener):
    def __init__(self, status: beam_eye_tracker.TrackingDataReceptionStatus):
        super().__init__()
        self.previous_status = status
        print_tracking_data_reception_status(status)

    def on_tracking_state_set_update(
        self, tracking_state_set: beam_eye_tracker.TrackingStateSet, timestamp: float
    ) -> None:
        print_latest_tracking_state_set(tracking_state_set, timestamp)

    def on_tracking_data_reception_status_changed(self, status: beam_eye_tracker.TrackingDataReceptionStatus) -> None:
        print_tracking_data_reception_status_if_changed(self.previous_status, status)
        self.previous_status = status


def main():
    # Initialize API with empty viewport geometry
    viewport_geometry = beam_eye_tracker.ViewportGeometry()
    viewport_geometry.point_00 = beam_eye_tracker.Point(0, 0)
    viewport_geometry.point_11 = beam_eye_tracker.Point(0, 0)

    bet_api = beam_eye_tracker.API("Python Async Sample", viewport_geometry)

    # Create an instance of your custom listener, which is the class receiving updates
    async_listener = MyTrackingListener(bet_api.get_tracking_data_reception_status())

    # Start receiving tracking data
    listener_handle = bet_api.start_receiving_tracking_data_on_listener(async_listener)

    # ... Do whatever you need to do ;-)
    time.sleep(30)

    bet_api.stop_receiving_tracking_data_on_listener(listener_handle)


if __name__ == "__main__":
    main()
