# Copyright (c) 2025, Eyeware Tech SA.
#
# All rights reserved.
#
# This sample demonstrates how to use the synchronous data access method.
#
# Synchronous data access is convenient in case you want to get a TrackingStateSet at low
# latency, and it is fine to block the thread while waiting for data updates.
#
# A typical scenario is when your application (or at least this thread) uses the tracking data
# updates as the event to trigger further processing/recording/interactions, etc.
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import bet_add_to_python_path

from eyeware import beam_eye_tracker
from bet_sample_utils import (
    print_tracking_data_reception_status,
    print_latest_tracking_state_set,
    print_tracking_data_reception_status_if_changed,
)


def main():
    # Initialize API with empty viewport geometry
    viewport_geometry = beam_eye_tracker.ViewportGeometry()
    viewport_geometry.point_00 = beam_eye_tracker.Point(0, 0)
    viewport_geometry.point_11 = beam_eye_tracker.Point(0, 0)

    bet_api = beam_eye_tracker.API("Python Sync Sample", viewport_geometry)

    # We hold a timestamp instance useful to sync the data reception
    last_update_timestamp_sec = beam_eye_tracker.NULL_DATA_TIMESTAMP()

    last_received_tracking_state_set = None
    previous_status = bet_api.get_tracking_data_reception_status()
    print_tracking_data_reception_status(previous_status)

    count = 0
    # Access 1 minute of data (assuming a 30fps webcam)
    while count < 30 * 60:
        wait_timeout_ms = 1000

        # Follow up with the status of the tracking data reception
        status = bet_api.get_tracking_data_reception_status()
        print_tracking_data_reception_status_if_changed(previous_status, status)
        previous_status = status

        # Wait for a new frame. If this function returns true, then there is a new frame and
        # last_update_timestamp_sec is updated
        if bet_api.wait_for_new_tracking_state_set(last_update_timestamp_sec, wait_timeout_ms):
            # This is how we access the latest TrackingStateSet
            last_received_tracking_state_set = bet_api.get_latest_tracking_state_set()
            print_latest_tracking_state_set(last_received_tracking_state_set, last_update_timestamp_sec)
            count += 1


if __name__ == "__main__":
    main()
