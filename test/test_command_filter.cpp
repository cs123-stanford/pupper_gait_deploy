// Host-side tests for the contract-v2 pieces the controller trusts blindly at
// 50 Hz: the obs_spec layout resolver, the command clip, and the IMU-yaw
// heading hold. The heading hold in particular is a transcription of mjlab's
// UniformVelocityCommand loop (velocity_command.py), and its failure modes --
// capturing on every step instead of the quiet edge, a truncating angle wrap,
// correcting at standstill -- are all silent until the robot veers.
//
// Deliberately standalone (no gtest), same as test_gait_reference.cpp: runs
// under ctest in the ROS build and under test/run_host_test.sh with nothing
// but a compiler.

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "json.hpp"
#include "neural_controller/command_filter.hpp"
#include "neural_controller/obs_spec.hpp"

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    failures++;
  }
}

void check_near(double got, double want, const char *what, double tol = 1e-9) {
  if (std::abs(got - want) > tol) {
    std::fprintf(stderr, "FAIL: %s (got %.6f, want %.6f)\n", what, got, want);
    failures++;
  }
}

void check_throws(void (*fn)(), const char *what) {
  try {
    fn();
    std::fprintf(stderr, "FAIL: %s did not throw\n", what);
    failures++;
  } catch (const std::exception &) {
  }
}

using neural_controller::CommandClip;
using neural_controller::HeadingHold;
using neural_controller::ObsLayout;
using neural_controller::legacy_obs_layout;
using neural_controller::parse_command_clip;
using neural_controller::parse_heading_hold;
using neural_controller::parse_obs_spec;
using neural_controller::wrap_to_pi;
using neural_controller::yaw_from_quaternion;

void test_wrap_to_pi() {
  check_near(wrap_to_pi(0.0), 0.0, "wrap_to_pi(0)");
  check_near(wrap_to_pi(3.0 * M_PI), M_PI, "wrap_to_pi(3pi)");
  check_near(wrap_to_pi(-0.5), -0.5, "wrap_to_pi(-0.5)");
  check_near(wrap_to_pi(M_PI + 0.1), -M_PI + 0.1, "wrap_to_pi(pi+0.1)");
  check_near(wrap_to_pi(-M_PI - 0.1), M_PI - 0.1, "wrap_to_pi(-pi-0.1)");
}

void test_yaw_from_quaternion() {
  // Pure yaw rotation by 0.7 rad: q = (cos(0.35), 0, 0, sin(0.35)).
  check_near(yaw_from_quaternion(std::cos(0.35), 0.0, 0.0, std::sin(0.35)), 0.7,
             "yaw of a pure-yaw quaternion");
  check_near(yaw_from_quaternion(1.0, 0.0, 0.0, 0.0), 0.0, "yaw of identity");
}

void test_command_clip() {
  CommandClip off;
  double vx = 9.0, vy = -9.0, wz = 9.0;
  off.apply(vx, vy, wz);
  check(vx == 9.0 && vy == -9.0 && wz == 9.0, "disabled clip must pass through");

  const nlohmann::json j = {{"vx", {-0.49, 0.49}}, {"vy", {-0.3, 0.3}}, {"wz", {-2.0, 2.0}}};
  CommandClip clip;
  parse_command_clip(j, clip);
  vx = 1.5;
  vy = -0.1;
  wz = -3.0;
  clip.apply(vx, vy, wz);
  check_near(vx, 0.49, "vx clamped to the cap");
  check_near(vy, -0.1, "in-range vy untouched");
  check_near(wz, -2.0, "wz clamped to the cap");

  // An unlisted axis stays unbounded.
  const nlohmann::json partial = {{"vx", {-0.5, 0.5}}};
  CommandClip clip_partial;
  parse_command_clip(partial, clip_partial);
  vy = 7.0;
  vx = 7.0;
  wz = 7.0;
  clip_partial.apply(vx, vy, wz);
  check(vx == 0.5 && vy == 7.0 && wz == 7.0, "unlisted axes stay unbounded");

  check_throws(
      [] {
        const nlohmann::json bad = {{"vx", {0.5, -0.5}}};
        CommandClip c;
        parse_command_clip(bad, c);
      },
      "inverted clip bounds");
}

HeadingHold make_hold() {
  const nlohmann::json j = {
      {"kp", 1.0}, {"clip", 0.3}, {"yaw_threshold", 0.1}, {"walk_threshold", 0.05}};
  HeadingHold hold;
  parse_heading_hold(j, hold);
  return hold;
}

void test_heading_hold_engages_and_corrects() {
  HeadingHold hold = make_hold();
  // Walking forward, quiet yaw: engages and captures heading 0.2.
  check_near(hold.apply(0.2, 0.5, 0.0, 0.0), 0.0, "on capture the error is zero");
  // Robot drifts to heading 0.1: correction is kp * (0.2 - 0.1).
  check_near(hold.apply(0.1, 0.5, 0.0, 0.0), 0.1, "P-correction toward the captured heading");
  // Large drift: correction clips at 0.3.
  check_near(hold.apply(-1.0, 0.5, 0.0, 0.0), 0.3, "correction clipped");
}

void test_heading_hold_captures_only_at_quiet_edge() {
  HeadingHold hold = make_hold();
  hold.apply(0.2, 0.5, 0.0, 0.0);  // Engage at heading 0.2.
  // Still engaged while the heading moves: the target must NOT follow it.
  check_near(hold.apply(0.3, 0.5, 0.0, 0.0), -0.1, "target held while engaged");
  check_near(hold.apply(0.4, 0.5, 0.0, 0.0), -0.2, "target still held");
}

void test_heading_hold_disengages_on_turn_and_recaptures() {
  HeadingHold hold = make_hold();
  hold.apply(0.2, 0.5, 0.0, 0.0);  // Engage at heading 0.2.
  // A commanded turn passes through raw and disengages.
  check_near(hold.apply(0.2, 0.5, 0.0, 1.5), 1.5, "commanded turn passes through");
  // Yaw quiet again: re-engage captures the NEW heading (0.9), so no correction.
  check_near(hold.apply(0.9, 0.5, 0.0, 0.0), 0.0, "re-engage captures the new heading");
}

void test_heading_hold_inactive_at_standstill() {
  HeadingHold hold = make_hold();
  // Not walking: raw command passes through, nothing captured.
  check_near(hold.apply(0.2, 0.0, 0.0, 0.0), 0.0, "standstill emits the raw yaw");
  check_near(hold.apply(0.2, 0.0, 0.0, 0.09), 0.09, "standstill passes raw yaw through");
  // Start walking at heading 0.5: the capture happens now, not earlier.
  check_near(hold.apply(0.5, 0.5, 0.0, 0.0), 0.0, "capture waits for the walk edge");
  check_near(hold.apply(0.4, 0.5, 0.0, 0.0), 0.1, "then corrects toward the walk-edge heading");
}

void test_heading_hold_wraps_across_pi() {
  HeadingHold hold = make_hold();
  hold.apply(M_PI - 0.05, 0.5, 0.0, 0.0);  // Engage just below +pi.
  // Heading crosses the seam to just above -pi: the error must wrap to -0.1,
  // not spin the long way around (which would clip at -0.3... after commanding
  // a full turn).
  check_near(hold.apply(-M_PI + 0.05, 0.5, 0.0, 0.0), -0.1, "heading error wraps across pi");
}

void test_heading_hold_disabled_passes_through() {
  HeadingHold hold;  // Not parsed from JSON: disabled.
  check_near(hold.apply(0.2, 0.5, 0.0, 0.7), 0.7, "disabled hold emits the raw yaw");
}

void test_obs_spec_v2_layout() {
  const nlohmann::json spec = {
      {"frame_order", "newest_first"},
      {"history", 20},
      {"components",
       {{{"name", "base_ang_vel"}, {"size", 3}},
        {{"name", "projected_gravity"}, {"size", 3}},
        {{"name", "command"}, {"size", 3}},
        {{"name", "desired_world_z"}, {"size", 3}},
        {{"name", "joint_pos_rel"}, {"size", 12}},
        {{"name", "last_action"}, {"size", 12}},
        {{"name", "gait_reference"}, {"size", 12}}}}};
  const ObsLayout layout = parse_obs_spec(spec, 12);
  check(layout.frame_size == 48, "v2 gait frame is 48");
  check(layout.ang_vel == 0 && layout.projected_gravity == 3 && layout.command == 6 &&
            layout.desired_world_z == 9 && layout.joint_pos_rel == 12 &&
            layout.last_action == 24 && layout.gait_reference == 36,
        "v2 offsets follow the component order");
}

void test_obs_spec_rejects_bad_specs() {
  check_throws(
      [] {
        const nlohmann::json spec = {
            {"frame_order", "newest_first"},
            {"components", {{{"name", "lidar"}, {"size", 16}}}}};
        parse_obs_spec(spec, 12);
      },
      "unknown component");
  check_throws(
      [] {
        const nlohmann::json spec = {
            {"frame_order", "newest_first"},
            {"components", {{{"name", "base_ang_vel"}, {"size", 4}}}}};
        parse_obs_spec(spec, 12);
      },
      "wrong component size");
  check_throws(
      [] {
        const nlohmann::json spec = {
            {"frame_order", "newest_first"},
            {"components",
             {{{"name", "command"}, {"size", 3}}, {{"name", "command"}, {"size", 3}}}}};
        parse_obs_spec(spec, 12);
      },
      "duplicate component");
  check_throws(
      [] {
        const nlohmann::json spec = {
            {"frame_order", "oldest_first"},
            {"components", {{{"name", "command"}, {"size", 3}}}}};
        parse_obs_spec(spec, 12);
      },
      "oldest_first frame order");
}

void test_legacy_layouts() {
  const ObsLayout base = legacy_obs_layout(36, 12);
  check(base.frame_size == 36 && base.gait_reference == -1, "legacy 36 frame");
  check(base.ang_vel == 0 && base.projected_gravity == 3 && base.command == 6 &&
            base.desired_world_z == 9 && base.joint_pos_rel == 12 && base.last_action == 24,
        "legacy 36 offsets");
  check_throws([] { legacy_obs_layout(48, 12); },
               "no obs_spec with a non-proprio frame size");
  check_throws([] { legacy_obs_layout(40, 12); }, "legacy layout with an unknown size");
}

}  // namespace

int main() {
  test_wrap_to_pi();
  test_yaw_from_quaternion();
  test_command_clip();
  test_heading_hold_engages_and_corrects();
  test_heading_hold_captures_only_at_quiet_edge();
  test_heading_hold_disengages_on_turn_and_recaptures();
  test_heading_hold_inactive_at_standstill();
  test_heading_hold_wraps_across_pi();
  test_heading_hold_disabled_passes_through();
  test_obs_spec_v2_layout();
  test_obs_spec_rejects_bad_specs();
  test_legacy_layouts();

  if (failures > 0) {
    std::fprintf(stderr, "\n%d check(s) FAILED\n", failures);
    return 1;
  }
  std::printf("OK: command filter, heading hold, and obs_spec checks passed\n");
  return 0;
}
