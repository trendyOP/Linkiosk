/**
 * Copyright (c) 2025, Eyeware Tech SA.
 *
 * All rights reserved.
 *
 * @brief This sample demonstrates how to auto start the Beam Eye Tracker.
 *
 */
#include "bet_sample_utils.h"
#include "eyeware/beam_eye_tracker.h"
#include <chrono>
#include <memory>
#include <thread>

using eyeware::beam_eye_tracker::TrackingDataReceptionStatus;
void main() {
    eyeware::beam_eye_tracker::API bet_api("Auto Start Beam Sample",
                                           eyeware::beam_eye_tracker::ViewportGeometry());

    TrackingDataReceptionStatus previous_status = bet_api.get_tracking_data_reception_status();
    print_tracking_data_reception_status(previous_status);

    bet_api.attempt_starting_the_beam_eye_tracker();

    // For now we will wait for status update changes.
    int count = 0;
    while (count < 400) {

        eyeware::beam_eye_tracker::TrackingDataReceptionStatus status =
            bet_api.get_tracking_data_reception_status();

        print_tracking_data_reception_status_if_changed(previous_status, status);

        if (previous_status == TrackingDataReceptionStatus::ATTEMPTING_TRACKING_AUTO_START &&
            status != previous_status) {
            // The attempt has concluded
            break;
        }
        previous_status = status;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        count++;
    }

    // Wait to let you read the output in console
    std::this_thread::sleep_for(std::chrono::seconds(5));
}