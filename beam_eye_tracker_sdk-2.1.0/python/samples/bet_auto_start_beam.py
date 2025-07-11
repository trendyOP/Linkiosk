# Copyright (c) 2025, Eyeware Tech SA.
#
# All rights reserved.
#
# This sample demonstrates how to auto start the Beam Eye Tracker.

import bet_add_to_python_path

import time
from eyeware import beam_eye_tracker as bet
from bet_sample_utils import (
    print_tracking_data_reception_status,
    print_tracking_data_reception_status_if_changed,
)


def main():
    # Initialize API with empty viewport geometry
    viewport_geom = bet.ViewportGeometry()
    viewport_geom.point_00 = bet.Point(0, 0)
    viewport_geom.point_11 = bet.Point(0, 0)

    bet_api = bet.API("Python Start Beam Sample", viewport_geom)

    previous_status = bet_api.get_tracking_data_reception_status()
    print_tracking_data_reception_status(previous_status)

    bet_api.attempt_starting_the_beam_eye_tracker()

    # For now we will wait for status update changes.
    count = 0
    while count < 400:
        status = bet_api.get_tracking_data_reception_status()
        print_tracking_data_reception_status_if_changed(previous_status, status)

        if (
            previous_status == bet.TrackingDataReceptionStatus.ATTEMPTING_TRACKING_AUTO_START
            and status != previous_status
        ):
            # The attempt has concluded
            break

        previous_status = status
        time.sleep(0.1)  # Sleep for 100ms
        count += 1

    # Wait to let you read the output in console
    time.sleep(5)


if __name__ == "__main__":
    main()
