// Parity test for trick (motion-tracking) policies' on-robot observations.
//
// motion_anchor_ori_b must equal what mjlab computed in training
// (subtract_frame_transforms + matrix_from_quat[..., :2]); a transposed or
// conjugated quaternion here would feed the policy a mirrored body orientation.
// trick_golden.json is generated from mjlab's own functions. Also checks the
// frame clock and the parser on a small motion block.
//
// Standalone (no gtest), like the other host tests.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "json.hpp"
#include "neural_controller/trick_motion.hpp"

using namespace neural_controller;

int main() {
  std::ifstream stream(TRICK_GOLDEN_JSON);
  if (!stream) throw std::runtime_error("could not open trick_golden.json");
  nlohmann::json golden;
  stream >> golden;
  int failures = 0;

  double worst = 0.0;
  for (const auto &c : golden["relative_orientation"]) {
    const Quat robot{c["robot_wxyz"][0], c["robot_wxyz"][1], c["robot_wxyz"][2], c["robot_wxyz"][3]};
    const Quat ref{c["reference_wxyz"][0], c["reference_wxyz"][1], c["reference_wxyz"][2],
                   c["reference_wxyz"][3]};
    float got[6];
    relative_orientation_6d(robot, ref, got);
    for (int k = 0; k < 6; k++) worst = std::max(worst, std::abs(got[k] - double(c["ori_6d"][k])));
  }
  std::printf("relative orientation: %zu cases, worst error %.2e\n",
              golden["relative_orientation"].size(), worst);
  if (worst > 1e-5) failures++;

  TrickMotion motion;
  motion.fps = golden["frame_at"]["fps"];
  motion.frames = golden["frame_at"]["frames"];
  for (const auto &c : golden["frame_at"]["cases"]) {
    const int got = motion.frame_at(double(c[0]));
    if (got != int(c[1])) {
      std::printf("frame_at(%.3f) = %d, expected %d\n", double(c[0]), got, int(c[1]));
      failures++;
    }
  }

  // Yaw alignment: rotating the reference by the captured offset makes the first
  // frame face the robot's heading, so the relative orientation starts at identity.
  const Quat robot_start = quat_mul(yaw_quat(1.1), Quat{0.9990, 0.0, -0.0447, 0.0});
  const Quat ref_start = yaw_quat(-0.4);
  const double offset = quat_yaw(robot_start) - quat_yaw(ref_start);
  float aligned[6];
  relative_orientation_6d(yaw_quat(quat_yaw(robot_start)), quat_mul(yaw_quat(offset), ref_start),
                          aligned);
  const float identity[6] = {1, 0, 0, 1, 0, 0};
  for (int k = 0; k < 6; k++) {
    if (std::abs(aligned[k] - identity[k]) > 1e-6) {
      std::printf("yaw alignment: component %d is %f, expected %f\n", k, aligned[k], identity[k]);
      failures++;
    }
  }

  nlohmann::json trick = {{"name", "t"}, {"fps", 50}};
  nlohmann::json block = {{"joint_pos", {std::vector<double>(12, 0.1), std::vector<double>(12, 0.2)}},
                          {"joint_vel", {std::vector<double>(12, 0.0), std::vector<double>(12, 1.0)}},
                          {"anchor_quat_wxyz", {{1, 0, 0, 0}, {1, 0, 0, 0}}}};
  parse_trick_motion(trick, block, 12, motion);
  if (motion.frames != 2 || std::abs(motion.joint_pos[12] - 0.2f) > 1e-6 ||
      std::abs(motion.joint_vel[23] - 1.0f) > 1e-6) {
    std::printf("parse_trick_motion: unexpected contents\n");
    failures++;
  }

  // Clip sequencing: A -> B (loops) ; requests C (-> B) and D (holds) ; a request
  // after the hold starts at once. 10 fps, clips share boundary frames.
  {
    nlohmann::json t = {{"name", "clips"}, {"fps", 10},
                        {"clips", {{{"name", "A"}, {"start", 0}, {"end", 10}, {"next", "B"}},
                                   {{"name", "B"}, {"start", 9}, {"end", 15}, {"loop", true}},
                                   {{"name", "C"}, {"start", 14}, {"end", 20}, {"next", "B"}},
                                   {{"name", "D"}, {"start", 19}, {"end", 25}}}}};
    std::vector<std::vector<double>> jp(25, std::vector<double>(12, 0.0));
    std::vector<std::vector<double>> aq(25, std::vector<double>{1, 0, 0, 0});
    nlohmann::json b = {{"joint_pos", jp}, {"joint_vel", jp}, {"anchor_quat_wxyz", aq}};
    TrickMotion m;
    parse_trick_motion(t, b, 12, m);
    ClipPlayer p;
    struct Step { double t; const char *request; int frame; const char *clip; };
    const Step steps[] = {
        {0.00, nullptr, 0, "A"},  {0.85, nullptr, 8, "A"},  {0.90, nullptr, 9, "B"},
        {1.20, "C", 12, "B"},     {1.40, nullptr, 14, "C"}, {1.85, nullptr, 18, "C"},
        {1.90, nullptr, 9, "B"},  {2.00, "D", 10, "B"},     {2.40, nullptr, 19, "D"},
        {2.95, nullptr, 24, "D"}, {3.40, nullptr, 24, "D"}, {3.50, "C", 14, "C"},
        {3.70, nullptr, 16, "C"},
    };
    for (const auto &st : steps) {
      if (st.request && !p.request(m, st.request)) {
        std::printf("clip request %s refused\n", st.request);
        failures++;
      }
      const int f = p.frame(m, st.t);
      const std::string clip = m.clips[p.clip()].name;
      if (f != st.frame || clip != st.clip) {
        std::printf("t=%.2f: frame %d in %s, expected %d in %s\n", st.t, f, clip.c_str(), st.frame,
                    st.clip);
        failures++;
      }
    }
    if (p.request(m, "nope")) {
      std::printf("unknown clip was accepted\n");
      failures++;
    }
  }

  std::printf(failures ? "FAILED (%d)\n" : "trick motion: all checks passed\n", failures);
  return failures ? 1 : 0;
}
