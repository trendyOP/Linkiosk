# Copyright (c) 2025, Eyeware Tech SA.
#
# All rights reserved.
#
# This script sets the Python path to the Beam Eye Tracker Python bindings.
# It is used to make the Beam Eye Tracker Python bindings available to the Python scripts in the
# samples/ directory. Whether you need to do this depends on your python setup and how you installed
# the Beam Eye Tracker Python API.

import os
import sys

# Hardcoded path to the Beam Eye Tracker Python bindings
BEAM_EYE_TRACKER_PYTHON_BINDINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "package")

# Set the Python path to the Beam Eye Tracker Python bindings
sys.path.append(BEAM_EYE_TRACKER_PYTHON_BINDINGS_PATH)
