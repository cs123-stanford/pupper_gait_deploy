#!/usr/bin/env bash
# Build and run the gait-reference parity test on a laptop -- no ROS, no gtest.
#
# Needs nlohmann/json's single header. It ships inside the monorepo's vendored
# RTNeural; point JSON_INCLUDE_DIR elsewhere if you are not next to a checkout.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

JSON_INCLUDE_DIR="${JSON_INCLUDE_DIR:-$HOME/projects/pupperv3-monorepo/ros2_ws/src/neural_controller/modules/RTNeural/modules/json}"
if [[ ! -f "$JSON_INCLUDE_DIR/json.hpp" ]]; then
  echo "json.hpp not found in $JSON_INCLUDE_DIR" >&2
  echo "Set JSON_INCLUDE_DIR to a directory containing nlohmann's json.hpp." >&2
  exit 1
fi

# The controller expects the header at neural_controller/gait_reference.hpp, which
# is where install.py puts it; stage that layout for the host build.
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
mkdir -p "$BUILD/include/neural_controller"
cp "$REPO/src/gait_reference.hpp" "$REPO/src/gait_reference_json.hpp" \
  "$REPO/src/command_filter.hpp" "$REPO/src/obs_spec.hpp" "$REPO/src/trick_motion.hpp" \
  "$BUILD/include/neural_controller/"

g++ -std=c++17 -O2 -Wall -Wextra \
  -I "$BUILD/include" -I "$JSON_INCLUDE_DIR" \
  -DGAIT_GOLDEN_JSON="\"$HERE/gait_golden.json\"" \
  -o "$BUILD/test_gait_reference" "$HERE/test_gait_reference.cpp"

g++ -std=c++17 -O2 -Wall -Wextra \
  -I "$BUILD/include" -I "$JSON_INCLUDE_DIR" \
  -o "$BUILD/test_command_filter" "$HERE/test_command_filter.cpp"

g++ -std=c++17 -O2 -Wall -Wextra \
  -I "$BUILD/include" -I "$JSON_INCLUDE_DIR" \
  -DTRICK_GOLDEN_JSON="\"$HERE/trick_golden.json\"" \
  -o "$BUILD/test_trick_motion" "$HERE/test_trick_motion.cpp"

"$BUILD/test_gait_reference" "$@"
"$BUILD/test_command_filter"
"$BUILD/test_trick_motion"
