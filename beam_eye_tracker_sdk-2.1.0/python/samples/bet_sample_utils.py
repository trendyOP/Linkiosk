# Copyright (c) 2025, Eyeware Tech SA.
#
# All rights reserved.
#
# This file demonstrates how to interpret the received data.
#
# Note: The timestamp fields are very informative within the API. They convey whether a data
# structure is valid at all (when holding NULL_DATA_TIMESTAMP) and whether "State" components  (eg.
# UserState, SimGameCameraState, etc.) are running at different framerates. These functions and
# comments are meant to provide guidance on how to interpret them. Accessing other data fields of a
# data structure with NULL_DATA_TIMESTAMP is undefined behavior.

from eyeware import beam_eye_tracker


def print_tracking_data_reception_status(status: beam_eye_tracker.TrackingDataReceptionStatus) -> None:
    """Prints a message indicating that no tracking data is being received.

    We just want this function to make the flow a bit more explicit in console.
    """
    print("****************************************************")
    if status == beam_eye_tracker.TrackingDataReceptionStatus.RECEIVING_TRACKING_DATA:
        print("Receiving tracking data.")
    elif status == beam_eye_tracker.TrackingDataReceptionStatus.NOT_RECEIVING_TRACKING_DATA:
        print("Not receiving tracking data.")
    elif status == beam_eye_tracker.TrackingDataReceptionStatus.ATTEMPTING_TRACKING_AUTO_START:
        print("Attempting to auto start tracking.")
    print("****************************************************")


def print_tracking_data_reception_status_if_changed(
    previous_status: beam_eye_tracker.TrackingDataReceptionStatus,
    status: beam_eye_tracker.TrackingDataReceptionStatus,
) -> None:
    """Shows how to interpret the tracking data reception status."""
    if previous_status == status:
        # No change in the status.
        return

    print("****************************************************")
    if status == beam_eye_tracker.TrackingDataReceptionStatus.RECEIVING_TRACKING_DATA:
        if previous_status == beam_eye_tracker.TrackingDataReceptionStatus.ATTEMPTING_TRACKING_AUTO_START:
            print("Successfully auto started the Beam Eye Tracker")
        else:
            print("Receiving tracking data.")
    elif status == beam_eye_tracker.TrackingDataReceptionStatus.NOT_RECEIVING_TRACKING_DATA:
        if previous_status == beam_eye_tracker.TrackingDataReceptionStatus.ATTEMPTING_TRACKING_AUTO_START:
            print("Failed to auto start the Beam Eye Tracker")
        else:
            print("Not receiving tracking data.")
    elif status == beam_eye_tracker.TrackingDataReceptionStatus.ATTEMPTING_TRACKING_AUTO_START:
        print("Attempting to auto start tracking.")
    print("****************************************************")


def print_user_state(user_state: beam_eye_tracker.UserState) -> None:
    """Shows how to interpret the user state."""
    if user_state.timestamp_in_seconds == beam_eye_tracker.NULL_DATA_TIMESTAMP():
        print("UserState data is not valid and should be ignored. ")
        return

    if user_state.head_pose.confidence == beam_eye_tracker.TrackingConfidence.LOST_TRACKING:
        # Scenario where the user face is not detected, the user goes away from frame, etc.
        # Warning: Using data from other fields of the UserState is undefined behavior.
        print("Tracking is active but the user is not being tracked. ")
        return
    else:
        print(
            f"Head pose: X = {user_state.head_pose.translation_from_hcs_to_wcs.x:.3f}, "
            f"Y = {user_state.head_pose.translation_from_hcs_to_wcs.y:.3f}, "
            f"Z = {user_state.head_pose.translation_from_hcs_to_wcs.z:.3f} ",
            end="",
        )

    if user_state.unified_screen_gaze.confidence == beam_eye_tracker.TrackingConfidence.LOST_TRACKING:
        # Screen gaze data is not being updated.
        print("User is not looking at the screen. ")
        return
    else:
        print(
            f"Point of regard: ({user_state.unified_screen_gaze.point_of_regard.x:.0f},"
            f"{user_state.unified_screen_gaze.point_of_regard.y:.0f}) "
        )


def print_latest_tracking_state_set(
    tracking_state_set: beam_eye_tracker.TrackingStateSet,
    timestamp_of_tracking_state_set: float,
) -> None:
    if timestamp_of_tracking_state_set == beam_eye_tracker.NULL_DATA_TIMESTAMP():
        # Typically, if following the sync, async or polling data access methods, it is unlikely to
        # end up reading a TrackingStateSet with an associated NULL_DATA_TIMESTAMP, but we still
        # include this case for completeness.
        print("Not receiving data from the Beam Eye Tracker. ")
        return

    # As of Beam 2.4.0, all "State" components (eg. UserState, SimGameCameraState, etc.) are
    # expected to be updated at the same time and thus holding the same timestamp as the
    # "timestamp_of_tracking_state_set" whenever the user is being tracked. However, a
    # future-proof implementation is to assume that only a subset the "State" components is
    # updated. One way of doing that, is to compare timestamps, as done in each case below.

    user_state = tracking_state_set.user_state()
    if (
        user_state.timestamp_in_seconds != beam_eye_tracker.NULL_DATA_TIMESTAMP()
        and user_state.timestamp_in_seconds != timestamp_of_tracking_state_set
    ):
        # This is the case when we keep receiving TrackingStateSet updates.
        print("UserState data not updated.")
    else:
        print_user_state(user_state)
