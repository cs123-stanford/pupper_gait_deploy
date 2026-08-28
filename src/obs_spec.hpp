#pragma once

#include <stdexcept>
#include <string>

namespace neural_controller {

/**
 * The observation frame layout, resolved from the policy JSON's "obs_spec"
 * block. Policies from the original lab pipeline have no obs_spec and get the
 * implicit plain proprio layout instead.
 *
 * The controller has a fixed menu of observation *providers* -- the quantities
 * it knows how to produce each control step -- and the policy declares which of
 * them it wants, in what order, as a list of named components. Each member
 * below is the offset of that component within a frame, or -1 when the policy
 * does not observe it. Retraining with a reordered, dropped, or re-sized
 * layout then deploys with no controller change; only a genuinely new
 * quantity means adding a provider here (and in neural_controller.cpp) once.
 *
 * Provider menu (sizes fixed by what the quantity is):
 *
 *   base_ang_vel(3)        IMU body angular velocity
 *   projected_gravity(3)   gravity direction in the body frame, from the IMU
 *   command(3)             the filtered (vx, vy, wz) teleop command
 *   desired_world_z(3)     commanded body-up direction (identity: [0, 0, 1])
 *   joint_pos_rel(12)      joint positions minus the default pose
 *   last_action(12)        the previous policy output
 *   gait_reference(12)     reference joint offset from the gait tables
 *   joint_vel_rel(12)      joint velocities
 *   motion_command(24)     a trick's reference joint angles (12) and velocities (12)
 *                          at the current frame
 *   motion_anchor_ori_b(6) a trick's reference body orientation in the robot's body
 *                          frame, first two rotation-matrix columns (see trick_motion.hpp)
 */
struct ObsLayout {
  int frame_size = 0;
  int ang_vel = -1;
  int projected_gravity = -1;
  int command = -1;
  int desired_world_z = -1;
  int joint_pos_rel = -1;
  int last_action = -1;
  int gait_reference = -1;
  int joint_vel_rel = -1;
  int motion_command = -1;
  int motion_anchor_ori_b = -1;
};

/** The implicit layout of original-pipeline policies (no obs_spec): the plain
 * 36-dim proprio frame. Anything else must declare its layout. */
inline ObsLayout legacy_obs_layout(int single_observation_size, int n_joints) {
  ObsLayout layout;
  layout.ang_vel = 0;
  layout.projected_gravity = 3;
  layout.command = 6;
  layout.desired_world_z = 9;
  layout.joint_pos_rel = 12;
  layout.last_action = 12 + n_joints;
  layout.frame_size = 12 + 2 * n_joints;
  if (single_observation_size != layout.frame_size) {
    throw std::runtime_error(
        "no obs_spec, and single_observation_size " +
        std::to_string(single_observation_size) +
        " is not the plain proprio frame; re-export the policy with mjlab");
  }
  return layout;
}

/**
 * Resolve the "obs_spec" block. Throws on an unknown component (the controller
 * cannot produce it), a size that disagrees with what the quantity is, a
 * duplicate, or a frame order other than newest_first (the only order the
 * history rotation implements).
 */
template <typename Json>
inline ObsLayout parse_obs_spec(const Json &spec, int n_joints) {
  const std::string order = spec.at("frame_order");
  if (order != "newest_first") {
    throw std::runtime_error("obs_spec.frame_order is '" + order +
                             "'; this controller only implements newest_first");
  }
  ObsLayout layout;
  auto place = [&](int &member, const std::string &name, int size, int expected) {
    if (size != expected) {
      throw std::runtime_error("obs_spec component " + name + " has size " +
                               std::to_string(size) + ", expected " +
                               std::to_string(expected));
    }
    if (member != -1) {
      throw std::runtime_error("obs_spec lists " + name + " twice");
    }
    member = layout.frame_size;
  };
  for (const auto &component : spec.at("components")) {
    const std::string name = component.at("name");
    const int size = component.at("size");
    if (name == "base_ang_vel") {
      place(layout.ang_vel, name, size, 3);
    } else if (name == "projected_gravity") {
      place(layout.projected_gravity, name, size, 3);
    } else if (name == "command") {
      place(layout.command, name, size, 3);
    } else if (name == "desired_world_z") {
      place(layout.desired_world_z, name, size, 3);
    } else if (name == "joint_pos_rel") {
      place(layout.joint_pos_rel, name, size, n_joints);
    } else if (name == "last_action") {
      place(layout.last_action, name, size, n_joints);
    } else if (name == "gait_reference") {
      place(layout.gait_reference, name, size, n_joints);
    } else if (name == "joint_vel_rel") {
      place(layout.joint_vel_rel, name, size, n_joints);
    } else if (name == "motion_command") {
      place(layout.motion_command, name, size, 2 * n_joints);
    } else if (name == "motion_anchor_ori_b") {
      place(layout.motion_anchor_ori_b, name, size, 6);
    } else {
      throw std::runtime_error("obs_spec component '" + name +
                               "' is not in this controller's provider menu");
    }
    layout.frame_size += size;
  }
  if (layout.frame_size <= 0) {
    throw std::runtime_error("obs_spec has no components");
  }
  return layout;
}

}  // namespace neural_controller
