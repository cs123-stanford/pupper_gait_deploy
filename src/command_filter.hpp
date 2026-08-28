#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace neural_controller {

/**
 * Deploy-side command conditioning, driven entirely by the policy JSON.
 *
 * Two stages, applied to the teleop command every policy step, in this order:
 *
 * 1. CommandClip -- clamp (vx, vy, wz) to the "command_clip" block. The
 *    exporter stamps the policy's trained command ranges by default, so the
 *    policy is never fed a command it has not seen; a tighter clip caps the
 *    deployed behavior below the trained band (e.g. shipping a mixed-gaits
 *    policy that only ever trots).
 *
 * 2. HeadingHold -- the IMU yaw P-loop from the "heading_hold" block, a direct
 *    transcription of mjlab's UniformVelocityCommand heading hold (see
 *    velocity_command.py): walking with a quiet commanded yaw, the heading at
 *    the moment the yaw went quiet is captured, and the emitted yaw command
 *    becomes clip(kp * heading_error, +-clip) until a turn is commanded or the
 *    linear command goes quiet. The policy trained against this exact
 *    closed-loop command profile, so the numbers ship with the export rather
 *    than being kept in sync by hand.
 *
 * The filtered command is what feeds the observation *and* the gait reference,
 * matching training, where the command manager's emitted buffer feeds both.
 */

inline double wrap_to_pi(double a) {
  // Direct transcription of mjlab's wrap_to_pi (MATLAB wrapToPi convention):
  // map to [-pi, pi], odd positive multiples of pi to +pi, negative to -pi.
  const double two_pi = 2.0 * M_PI;
  double wrapped = std::fmod(a + M_PI, two_pi);
  if (wrapped < 0.0) {
    wrapped += two_pi;  // Floor-mod, [0, 2pi), like torch's %.
  }
  if (wrapped == 0.0 && a > 0.0) {
    return M_PI;
  }
  return wrapped - M_PI;
}

// Yaw (heading) of the body from the IMU orientation quaternion.
inline double yaw_from_quaternion(double w, double x, double y, double z) {
  return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

struct CommandClip {
  bool enabled = false;
  // Axes absent from the JSON stay unbounded.
  double vx_lo = -std::numeric_limits<double>::infinity();
  double vx_hi = std::numeric_limits<double>::infinity();
  double vy_lo = -std::numeric_limits<double>::infinity();
  double vy_hi = std::numeric_limits<double>::infinity();
  double wz_lo = -std::numeric_limits<double>::infinity();
  double wz_hi = std::numeric_limits<double>::infinity();

  void apply(double &vx, double &vy, double &wz) const {
    if (!enabled) {
      return;
    }
    vx = std::min(std::max(vx, vx_lo), vx_hi);
    vy = std::min(std::max(vy, vy_lo), vy_hi);
    wz = std::min(std::max(wz, wz_lo), wz_hi);
  }
};

struct HeadingHold {
  bool enabled = false;
  double kp = 0.0;
  double clip = 0.0;
  double yaw_threshold = 0.0;
  double walk_threshold = 0.0;

  // Loop state.
  double target = 0.0;
  bool prev_active = false;

  void reset() {
    target = 0.0;
    prev_active = false;
  }

  /**
   * The yaw command the policy should see: the clipped P-correction while the
   * hold is engaged, the raw command otherwise. `heading` is the IMU yaw.
   */
  double apply(double heading, double vx, double vy, double raw_yaw) {
    if (!enabled) {
      return raw_yaw;
    }
    const bool active = std::abs(raw_yaw) < yaw_threshold &&
                        std::sqrt(vx * vx + vy * vy) > walk_threshold;
    if (active && !prev_active) {
      target = heading;  // Capture at the quiet edge.
    }
    prev_active = active;
    if (!active) {
      return raw_yaw;
    }
    const double err = wrap_to_pi(target - heading);
    return std::min(std::max(kp * err, -clip), clip);
  }
};

/**
 * Fill a CommandClip from the "command_clip" block of a deploy JSON. Templated
 * on the JSON type for the same reason as parse_gait_reference: nlohmann::json
 * is only reachable through RTNeural's vendored copy, and the host-side test
 * includes the header directly.
 */
template <typename Json>
inline void parse_command_clip(const Json &block, CommandClip &clip) {
  auto read_pair = [&](const char *key, double &lo, double &hi) {
    if (!block.contains(key)) {
      return;  // Unlisted axes stay unbounded.
    }
    const auto &pair = block.at(key);
    lo = pair.at(0);
    hi = pair.at(1);
    if (lo > hi) {
      throw std::runtime_error(std::string("command_clip.") + key + " is inverted");
    }
  };
  read_pair("vx", clip.vx_lo, clip.vx_hi);
  read_pair("vy", clip.vy_lo, clip.vy_hi);
  read_pair("wz", clip.wz_lo, clip.wz_hi);
  clip.enabled = true;
}

/** Fill a HeadingHold from the "heading_hold" block of a deploy JSON. */
template <typename Json>
inline void parse_heading_hold(const Json &block, HeadingHold &hold) {
  hold.kp = block.at("kp");
  hold.clip = block.at("clip");
  hold.yaw_threshold = block.at("yaw_threshold");
  hold.walk_threshold = block.at("walk_threshold");
  hold.enabled = true;
  hold.reset();
}

}  // namespace neural_controller
