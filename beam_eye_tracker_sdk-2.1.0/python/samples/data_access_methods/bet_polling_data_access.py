# Copyright (c) 2025, Eyeware Tech SA.
#
# All rights reserved.
#
# This sample demonstrates how to use the polling data access method.
#
# Polling is convenient in case your thread can't be blocked, and it is not critical to read the
# TrackingStateSet as soon as it is available, meaning, you can afford some latency.
#
# A typical scenario is when your thread loop is driven by other events and you want to use
# the latest TrackingStateSet in combination with said events. For example, in a game, where
# there is a logic driven by frames rendering: you don't want to block, but you want to use the
# eye tracking for some interaction in the game.
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


def main():
    # Initialize API with empty viewport geometry
    viewport_geometry = beam_eye_tracker.ViewportGeometry()
    viewport_geometry.point_00 = beam_eye_tracker.Point(0, 0)
    viewport_geometry.point_11 = beam_eye_tracker.Point(0, 0)

    bet_api = beam_eye_tracker.API("Python Polling Sample", viewport_geometry)

    # We hold a timestamp instance useful to sync the data reception
    last_update_timestamp_sec = beam_eye_tracker.NULL_DATA_TIMESTAMP()

    last_received_tracking_state_set = None
    previous_status = bet_api.get_tracking_data_reception_status()
    print_tracking_data_reception_status(previous_status)

    count = 0
    # Access 1 minute of data (assuming a 10fps events)
    while count < 10 * 60:
        # Follow up with the status of the tracking data reception
        status = bet_api.get_tracking_data_reception_status()
        print_tracking_data_reception_status_if_changed(previous_status, status)
        previous_status = status

        # Polling for new data follows the synchronous data access method, but with a timeout of
        # 0ms. This means that the function will return immediately.
        # Similarly, last_update_timestamp_sec is updated in case of new data.
        if bet_api.wait_for_new_tracking_state_set(last_update_timestamp_sec, 0):
            # This is how we access the latest TrackingStateSet
            last_received_tracking_state_set = bet_api.get_latest_tracking_state_set()
            print_latest_tracking_state_set(last_received_tracking_state_set, last_update_timestamp_sec)
            count += 1

        # To simulate an "external event" driving this thread at 10fps, we sleep 100ms to simulate
        # a thread loop driven by other events.
        time.sleep(0.1)


if __name__ == "__main__":
    main()
