#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "neural_controller/gait_reference.hpp"

namespace neural_controller {

/**
 * Fill a GaitReference from the "gait_reference" block of a deploy JSON.
 *
 * Templated on the JSON type so this header needs no include of its own: RTNeural
 * vendors nlohmann/json behind a relative path, so `nlohmann::json` is only
 * reachable to translation units that already pulled in RTNeural.h. The host-side
 * parity test includes the vendored header directly and instantiates this the same
 * way, which is the point -- the controller and the test share one parser.
 *
 * Throws std::runtime_error on anything malformed (a missing field, or a table whose
 * shape disagrees with n_samples) so a bad policy fails at load, not mid-stride.
 *
 * @param block    The "gait_reference" object itself, not the whole policy.
 * @param n_joints Number of actuated joints the controller drives.
 */
template <typename Json>
inline void parse_gait_reference(const Json &block, int n_joints, GaitReference &gait) {
  gait.n_joints = n_joints;
  gait.n_samples = block.at("n_samples");
  gait.frequency = block.at("frequency");
  gait.blend_speed = block.at("blend_speed");
  gait.gallop_speed = block.at("gallop_speed");
  gait.gallop_freq_mult = block.at("gallop_freq_mult");
  gait.dir_threshold = block.at("dir_threshold");

  auto read_table = [&](const std::string &key, std::vector<double> &table) {
    const auto &rows = block.at(key);
    if (static_cast<int>(rows.size()) != gait.n_samples) {
      throw std::runtime_error(key + " has " + std::to_string(rows.size()) +
                               " rows, expected n_samples=" + std::to_string(gait.n_samples));
    }
    table.clear();
    table.reserve(static_cast<std::size_t>(gait.n_samples) * n_joints);
    for (const auto &row : rows) {
      if (static_cast<int>(row.size()) != n_joints) {
        throw std::runtime_error(key + " row has " + std::to_string(row.size()) +
                                 " entries, expected " + std::to_string(n_joints));
      }
      for (const auto &v : row) {
        table.push_back(v);
      }
    }
  };
  read_table("trot_table", gait.trot_table);
  read_table("gallop_table", gait.gallop_table);
  // Mixed-gait policies also ship a lift-in-place table; older JSONs don't.
  if (block.contains("lift_table")) {
    read_table("lift_table", gait.lift_table);
  }
  // Direction-split fast gaits ship a separate backward capture; older JSONs
  // time-reverse the single fast table instead.
  if (block.contains("gallop_back_table")) {
    read_table("gallop_back_table", gait.gallop_back_table);
  }

  if (!gait.valid()) {
    throw std::runtime_error("gait_reference block is not self-consistent");
  }
}

/**
 * Fill a JumpSlot from the "jump_slot" block of a MixedGaitsJump deploy JSON.
 */
template <typename Json>
inline void parse_jump_slot(const Json &block, int n_joints, JumpSlot &slot) {
  slot.n_joints = n_joints;
  slot.n_samples = block.at("n_samples");
  slot.playback_s = block.at("playback_s");
  slot.active_s = block.at("active_s");
  slot.cross_fade_s = block.at("cross_fade_s");
  slot.grid_s = block.at("grid_s");
  slot.busy_s = block.at("busy_s");
  if (block.contains("trigger_button")) slot.trigger_button = block.at("trigger_button");
  if (block.contains("run_button")) slot.run_button = block.at("run_button");
  if (block.contains("walk_speed_cap")) slot.walk_speed_cap = block.at("walk_speed_cap");
  if (block.contains("run_speed_cap")) slot.run_speed_cap = block.at("run_speed_cap");

  const auto &rows = block.at("jump_table");
  if (static_cast<int>(rows.size()) != slot.n_samples) {
    throw std::runtime_error("jump_slot table has " + std::to_string(rows.size()) +
                             " rows, expected n_samples=" + std::to_string(slot.n_samples));
  }
  slot.jump_table.clear();
  slot.jump_table.reserve(static_cast<std::size_t>(slot.n_samples) * n_joints);
  for (const auto &row : rows) {
    if (static_cast<int>(row.size()) != n_joints) {
      throw std::runtime_error("jump_slot row has " + std::to_string(row.size()) +
                               " entries, expected " + std::to_string(n_joints));
    }
    for (const auto &v : row) {
      slot.jump_table.push_back(v);
    }
  }
  // Optional running hop (newer exports; absent = jump only).
  slot.hop_table.clear();
  if (block.contains("hop_table")) {
    const auto &hop_rows = block.at("hop_table");
    slot.hop_n_samples = static_cast<int>(hop_rows.size());
    slot.hop_speed = block.at("hop_speed");
    slot.hop_play_s = block.at("hop_play_s");
    slot.hop_table_s = block.at("hop_table_s");
    slot.hop_reach_frequency = block.at("hop_reach_frequency");
    slot.hop_table.reserve(static_cast<std::size_t>(slot.hop_n_samples) * n_joints);
    for (const auto &row : hop_rows) {
      if (static_cast<int>(row.size()) != n_joints) {
        throw std::runtime_error("jump_slot hop row has " + std::to_string(row.size()) +
                                 " entries, expected " + std::to_string(n_joints));
      }
      for (const auto &v : row) {
        slot.hop_table.push_back(v);
      }
    }
  }
  if (!slot.valid()) {
    throw std::runtime_error("jump_slot block is not self-consistent");
  }
}

}  // namespace neural_controller
