#
# Copyright (C) 2025 Eyeware Tech SA.
#
cmake_minimum_required(VERSION 3.15)
project(beam_game_engine_integration_sample)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Hard-code here where you unzipped the Beam Eye Tracker SDK.
set(BEAM_EYE_TRACKER_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." CACHE PATH "Path to beam_eye_tracker_client installation")
set(PLATFORM "win64")

# Verify the required directories exist
if(NOT EXISTS "${BEAM_EYE_TRACKER_SDK_DIR}")
    message(FATAL_ERROR "beam_eye_tracker_client directory not found at ${BEAM_EYE_TRACKER_SDK_DIR}")
endif()

######################### BEAM EYE TRACKER LIBRARY #########################
# Create interface library for Beam Eye Tracker SDK
add_library(beam_eye_tracker_client INTERFACE)
set_target_properties(beam_eye_tracker_client PROPERTIES
    INTERFACE_DLL_LOCATION "${BEAM_EYE_TRACKER_SDK_DIR}/bin/${PLATFORM}/beam_eye_tracker_client.dll"
)
target_include_directories(beam_eye_tracker_client
    INTERFACE
        ${BEAM_EYE_TRACKER_SDK_DIR}/include
)
target_link_libraries(beam_eye_tracker_client
    INTERFACE
        "${BEAM_EYE_TRACKER_SDK_DIR}/lib/${PLATFORM}/beam_eye_tracker_client.lib"
)

# Install the DLL next to samples executables.
get_target_property(DLL_LOCATION beam_eye_tracker_client INTERFACE_DLL_LOCATION)
install(FILES "${DLL_LOCATION}" DESTINATION ./${PLATFORM})

function(add_beam_sample TARGET_NAME SAMPLE_SOURCE_FILES   )
    add_executable(${TARGET_NAME} ${SAMPLE_SOURCE_FILES})
    target_link_libraries(${TARGET_NAME} PRIVATE beam_eye_tracker_client)
    install(TARGETS ${TARGET_NAME} DESTINATION ./${PLATFORM})
endfunction()

################ GAME ENGINE INTEGRATION SAMPLE #################
add_beam_sample(game_engine_integration
    game_engine_integration/main.cpp
    game_engine_integration/bet_game_engine_device.h
    game_engine_integration/my_game_engine.h
)
################ DATA ACCESS METHODS SAMPLES #################
add_beam_sample(sync_data_access
    data_access_methods/bet_sync_data_access.cpp
)
add_beam_sample(polling_data_access
    data_access_methods/bet_polling_data_access.cpp
)
add_beam_sample(async_data_access
    data_access_methods/bet_async_data_access.cpp
)

################ AUTO START BEAM SAMPLE #################
add_beam_sample(auto_start_beam bet_auto_start_beam.cpp)


