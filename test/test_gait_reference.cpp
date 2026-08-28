// Cross-language parity test for the on-robot gait reference.
//
// The reference joint offset the controller feeds the policy has to match, to
// within float noise, the one mjlab computed during training -- a phase-wrap or
// modulo slip here shows up on hardware as a policy that has never seen its own
// input. gait_golden.json holds tables and expected offsets generated straight
// from the trained env by mjlab's `export-pupper-policy --golden-output`, covering
// standing, trot, gallop, and the time-reversed backward / turning cases.
//
// argv[1] is the golden fixture; argv[2] is optionally a deploy policy.json, whose
// gait_reference block is then parsed through the same parser the controller uses
// and checked against the fixture's tables.
//
// Deliberately standalone (no gtest): it runs under ctest in the ROS build and
// under test/run_host_test.sh with nothing but a compiler.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"
#include "neural_controller/gait_reference_json.hpp"

namespace {

constexpr double kTolerance = 1e-4;
constexpr int kActionSize = 12;

nlohmann::json read_json(const std::string &path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("could not open " + path);
  }
  nlohmann::json j;
  stream >> j;
  return j;
}

}  // namespace

int main(int argc, char **argv) {
  const std::string path = argc > 1 ? argv[1] : GAIT_GOLDEN_JSON;
  // Optional: a real deploy policy.json, to confirm its gait_reference block
  // parses and carries the same tables the golden data was generated with.
  const std::string policy_path = argc > 2 ? argv[2] : "";

  nlohmann::json j;
  neural_controller::GaitReference gait;
  try {
    j = read_json(path);
    neural_controller::parse_gait_reference(j.at("gait_reference"), kActionSize, gait);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }

  if (!policy_path.empty()) {
    std::ifstream probe(policy_path);
    const bool present = static_cast<bool>(probe);
    probe.close();
    if (!present) {
      // Nothing deployed here yet (download_policy.py has not run); the golden
      // checks below still cover the math.
      std::printf("No policy at %s -- skipping the deployed-policy check\n",
                  policy_path.c_str());
    } else {
      try {
        const nlohmann::json policy = read_json(policy_path);
        neural_controller::GaitReference from_policy;
        neural_controller::parse_gait_reference(policy.at("gait_reference"), kActionSize,
                                                from_policy);
        const int frame = policy.value("single_observation_size", 36);
        if (frame != 36 + kActionSize) {
          std::fprintf(stderr, "FAIL: %s has gait_reference but frame size %d\n",
                       policy_path.c_str(), frame);
          return 1;
        }
        // Matching tables mean the deployed policy came from the same task
        // config the golden was generated against. A mismatch is normal (any
        // other run/config deploys tables of its own) -- the math checks below
        // still pin the lookup down, so it is informational, not a failure.
        if (from_policy.trot_table != gait.trot_table ||
            from_policy.gallop_table != gait.gallop_table ||
            from_policy.lift_table != gait.lift_table ||
            from_policy.gallop_back_table != gait.gallop_back_table) {
          std::printf(
              "Deploy policy %s: gait_reference parsed; tables differ from the golden "
              "fixture (different training config) -- golden math checks still apply\n",
              policy_path.c_str());
        } else {
          std::printf("Deploy policy %s: gait_reference parsed, tables match the golden data\n",
                      policy_path.c_str());
        }
      } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: parsing %s: %s\n", policy_path.c_str(), e.what());
        return 1;
      }
    }
  }

  std::vector<double> default_joint_pos = j.at("default_joint_pos").get<std::vector<double>>();
  if (static_cast<int>(default_joint_pos.size()) != gait.n_joints) {
    std::fprintf(stderr, "FAIL: default_joint_pos has %zu entries, expected %d\n",
                 default_joint_pos.size(), gait.n_joints);
    return 1;
  }

  int failures = 0;
  double worst = 0.0;
  int n_cases = 0;
  std::vector<float> out(gait.n_joints, 0.0f);

  for (const auto &c : j.at("cases")) {
    const double t = c.at("t");
    const double vx = c.at("vx");
    const double vy = c.at("vy");
    const double yaw = c.at("yaw");
    const std::vector<double> expected = c.at("expected").get<std::vector<double>>();

    neural_controller::compute_gait_reference_offset(gait, default_joint_pos, t, vx, vy, yaw,
                                                     out.data());

    double max_err = 0.0;
    for (int i = 0; i < gait.n_joints; i++) {
      max_err = std::max(max_err, std::abs(static_cast<double>(out[i]) - expected[i]));
    }
    worst = std::max(worst, max_err);
    n_cases++;
    if (max_err > kTolerance) {
      std::fprintf(stderr,
                   "FAIL case t=%.3f vx=%+.3f vy=%+.3f yaw=%+.3f: max_abs_err=%.3e\n", t, vx,
                   vy, yaw, max_err);
      failures++;
    }
  }

  // A zero command must produce an all-zero offset: the policy reads that as
  // "stand at the default pose", and it is what the controller's freshly
  // initialized observation history contains.
  neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 3.7, 0.0, 0.0, 0.0,
                                                   out.data());
  for (int i = 0; i < gait.n_joints; i++) {
    if (out[i] != 0.0f) {
      std::fprintf(stderr, "FAIL: zero command gave a nonzero offset at joint %d (%g)\n", i,
                   static_cast<double>(out[i]));
      failures++;
      break;
    }
  }

  // Backward must not equal forward at the same phase: that is the time reversal,
  // and it is exactly what a truncating modulo would silently break.
  std::vector<float> fwd(gait.n_joints), bwd(gait.n_joints);
  neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 0.31, 0.3, 0.0, 0.0,
                                                   fwd.data());
  neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 0.31, -0.3, 0.0, 0.0,
                                                   bwd.data());
  bool differs = false;
  for (int i = 0; i < gait.n_joints; i++) {
    differs = differs || std::abs(fwd[i] - bwd[i]) > 1e-3f;
  }
  if (!differs) {
    std::fprintf(stderr, "FAIL: backward reference matches forward -- time reversal lost\n");
    failures++;
  }

  // Crossing gallop_speed must switch tables.
  std::vector<float> trot(gait.n_joints), gallop(gait.n_joints);
  neural_controller::compute_gait_reference_offset(
      gait, default_joint_pos, 0.4, gait.gallop_speed - 1e-3, 0.0, 0.0, trot.data());
  neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 0.4,
                                                   gait.gallop_speed, 0.0, 0.0, gallop.data());
  differs = false;
  for (int i = 0; i < gait.n_joints; i++) {
    differs = differs || std::abs(trot[i] - gallop[i]) > 1e-3f;
  }
  if (!differs) {
    std::fprintf(stderr, "FAIL: gallop threshold did not switch the reference table\n");
    failures++;
  }

  // Mixed-gait policies: a pure turn must play the lift table, not the trot.
  // Compare against the same gait with the lift table dropped (the legacy path).
  int property_checks = 3;
  if (!gait.lift_table.empty()) {
    property_checks++;
    neural_controller::GaitReference no_lift = gait;
    no_lift.lift_table.clear();
    std::vector<float> lifted(gait.n_joints), trotted(gait.n_joints);
    neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 0.4, 0.0, 0.0,
                                                     0.9, lifted.data());
    neural_controller::compute_gait_reference_offset(no_lift, default_joint_pos, 0.4, 0.0,
                                                     0.0, 0.9, trotted.data());
    differs = false;
    for (int i = 0; i < gait.n_joints; i++) {
      differs = differs || std::abs(lifted[i] - trotted[i]) > 1e-3f;
    }
    if (!differs) {
      std::fprintf(stderr, "FAIL: turning ignored the lift table\n");
      failures++;
    }
  }

  // Direction-split policies: a backward-fast command must play the backward
  // capture, not the time-reversed forward table (which is what dropping the
  // table silently falls back to).
  if (!gait.gallop_back_table.empty() && gait.gallop_back_table != gait.gallop_table) {
    property_checks++;
    neural_controller::GaitReference merged = gait;
    merged.gallop_back_table.clear();
    std::vector<float> split(gait.n_joints), fallback(gait.n_joints);
    const double fast_vx = -(gait.gallop_speed + 0.2);
    neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 0.4, fast_vx,
                                                     0.0, 0.0, split.data());
    neural_controller::compute_gait_reference_offset(merged, default_joint_pos, 0.4, fast_vx,
                                                     0.0, 0.0, fallback.data());
    differs = false;
    for (int i = 0; i < gait.n_joints; i++) {
      differs = differs || std::abs(split[i] - fallback[i]) > 1e-3f;
    }
    if (!differs) {
      std::fprintf(stderr, "FAIL: backward-fast ignored the gallop_back table\n");
      failures++;
    }
    // And the forward-fast branch must not be perturbed by its presence.
    neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 0.4, -fast_vx,
                                                     0.0, 0.0, split.data());
    neural_controller::compute_gait_reference_offset(merged, default_joint_pos, 0.4, -fast_vx,
                                                     0.0, 0.0, fallback.data());
    for (int i = 0; i < gait.n_joints; i++) {
      if (split[i] != fallback[i]) {
        std::fprintf(stderr, "FAIL: gallop_back table perturbed the forward-fast branch\n");
        failures++;
        break;
      }
    }
  }

  // MixedGaitsJump fixtures: the jump-slot composite cases.
  if (j.contains("jump_slot")) {
    neural_controller::JumpSlot slot;
    try {
      neural_controller::parse_jump_slot(j.at("jump_slot"), kActionSize, slot);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "FAIL: %s\n", e.what());
      return 1;
    }
    property_checks++;
    double slot_worst = 0.0;
    int slot_cases = 0;
    for (const auto &c : j.at("slot_cases")) {
      const double t = c.at("t");
      const std::vector<double> expected = c.at("expected").get<std::vector<double>>();
      // Newer fixtures mark hop cases (the latched choice); older ones are jumps.
      const bool hop = c.contains("hop") && c.at("hop").template get<bool>();
      neural_controller::compute_mixed_jump_reference_offset(
          gait, slot, default_joint_pos, t, c.at("vx"), c.at("vy"), c.at("yaw"),
          c.at("slot_start"), hop, out.data());
      double max_err = 0.0;
      for (int i = 0; i < gait.n_joints; i++) {
        max_err = std::max(max_err, std::abs(static_cast<double>(out[i]) - expected[i]));
      }
      slot_worst = std::max(slot_worst, max_err);
      slot_cases++;
      if (max_err > kTolerance) {
        std::fprintf(stderr, "FAIL slot case t=%.3f: max_abs_err=%.3e\n", t, max_err);
        failures++;
      }
    }
    // No slot (infinite start) must reproduce the plain mixed reference bit-for-bit.
    std::vector<float> plain(gait.n_joints), composite(gait.n_joints);
    neural_controller::compute_gait_reference_offset(gait, default_joint_pos, 2.3, 0.4, 0.0,
                                                     0.0, plain.data());
    neural_controller::compute_mixed_jump_reference_offset(
        gait, slot, default_joint_pos, 2.3, 0.4, 0.0, 0.0,
        std::numeric_limits<double>::infinity(), composite.data());
    for (int i = 0; i < gait.n_joints; i++) {
      if (plain[i] != composite[i]) {
        std::fprintf(stderr, "FAIL: slot-free composite differs from the mixed reference\n");
        failures++;
        break;
      }
    }
    std::printf("jump_slot: %d composite cases matched (worst %.3e)\n", slot_cases,
                slot_worst);
  }

  if (failures > 0) {
    std::fprintf(stderr, "\n%d check(s) FAILED\n", failures);
    return 1;
  }
  std::printf("OK: %d golden cases matched (worst max_abs_err = %.3e) + %d property checks\n",
              n_cases, worst, property_checks);
  return 0;
}
