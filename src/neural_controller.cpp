#include "neural_controller/neural_controller.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "controller_interface/helpers.hpp"
#include "hardware_interface/loaned_command_interface.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/qos.hpp"

namespace neural_controller {
NeuralController::NeuralController()
    : controller_interface::ControllerInterface(),
      rt_cmd_vel_ptr_(nullptr),
      rt_cmd_pose_ptr_(nullptr) {}

// Check parameter vectors have the correct size
bool NeuralController::check_param_vector_size() {
  const std::vector<std::pair<std::string, size_t>> param_sizes = {
      {"action_scales", params_.action_scales.size()},
      {"action_types", params_.action_types.size()},
      {"kps", params_.kps.size()},
      {"kds", params_.kds.size()},
      {"kp_scale", params_.kp_scale.size()},
      {"init_kps", params_.init_kps.size()},
      {"init_kds", params_.init_kds.size()},
      {"default_joint_pos", params_.default_joint_pos.size()},
      {"joint_lower_limits", params_.joint_lower_limits.size()},
      {"joint_upper_limits", params_.joint_upper_limits.size()},
      {"joint_names", params_.joint_names.size()}};

  for (const auto &[name, size] : param_sizes) {
    if (size != kActionSize) {
      RCLCPP_ERROR(get_node()->get_logger(), "%s size is %ld, expected %d", name.c_str(), size,
                   kActionSize);
      return false;
    }
  }
  return true;
}

controller_interface::CallbackReturn NeuralController::on_init() {
  try {
    param_listener_ = std::make_shared<ParamListener>(get_node());
    params_ = param_listener_->get_params();

    if (params_.gain_multiplier < 0.0) {
      RCLCPP_ERROR(get_node()->get_logger(), "Gain_multiplier must be >= 0.0. Stopping");
      return controller_interface::CallbackReturn::ERROR;
    }
    if (params_.gain_multiplier != 1.0) {
      RCLCPP_WARN(get_node()->get_logger(), "Gain_multiplier is set to %f",
                  params_.gain_multiplier);
    }

    // Per-joint kp trim, for hardware variation between individual robots -- a
    // motor that is weak relative to the rest needs a stiffer loop to track the
    // same command. This multiplies the kp the policy asks for, rather than
    // replacing it, because the policy JSON stamps kp as a single scalar over
    // all twelve joints (see set_param_from_json_scalar below), so it would
    // overwrite anything per-joint set in kps. Left out of a config entirely,
    // it means "no trim".
    if (params_.kp_scale.empty()) {
      params_.kp_scale.assign(kActionSize, 1.0);
    }
    for (const auto &scale : params_.kp_scale) {
      if (scale < 0.0) {
        RCLCPP_ERROR(get_node()->get_logger(), "kp_scale entries must be >= 0.0. Stopping");
        return controller_interface::CallbackReturn::ERROR;
      }
    }

    std::ifstream json_stream(params_.model_path, std::ifstream::binary);
    model_ = RTNeural::json_parser::parseJson<float>(json_stream, true);

    // Read params json file using nholsojson to extract metadata
    nlohmann::json j;
    std::ifstream json_file(params_.model_path);
    json_file >> j;

    auto set_param_from_json_vector = [&](const std::string &key, auto &param) {
      if (j.find(key) != j.end()) {
        RCLCPP_INFO(get_node()->get_logger(), "From JSON, setting %s vector element-by-element",
                    key.c_str());
        if (j[key].size() != kActionSize) {
          std::string error_msg = "Invalid size for " + key + " (" + std::to_string(j[key].size()) +
                                  ") != " + std::to_string(kActionSize);
          RCLCPP_ERROR(get_node()->get_logger(), "%s", error_msg.c_str());
          throw std::runtime_error(error_msg);
        }
        param.resize(j[key].size(), 0.0);
        for (int i = 0; i < param.size(); i++) {
          param.at(i) = j[key].at(i);
        }
      }
    };

    auto set_param_from_json_scalar = [&](const std::string &key, auto &param, int size) {
      if (j.find(key) != j.end()) {
        RCLCPP_INFO(get_node()->get_logger(), "From JSON, setting %s[:]=%f", key.c_str(),
                    static_cast<double>(j[key]));
        param.resize(size, 0.0);
        for (auto &p : param) {
          p = j[key];
        }
      }
    };

    auto set_param_from_json_mixed = [&](const std::string &key, auto &param, int size) {
      if (j.find(key) != j.end()) {
        if (j[key].is_array()) {
          set_param_from_json_vector(key, param);
        } else {
          set_param_from_json_scalar(key, param, size);
        }
      }
    };

    set_param_from_json_scalar("kp", params_.kps, kActionSize);
    set_param_from_json_scalar("kd", params_.kds, kActionSize);
    set_param_from_json_mixed("action_scale", params_.action_scales, kActionSize);
    set_param_from_json_vector("default_joint_pos", params_.default_joint_pos);
    set_param_from_json_vector("joint_lower_limits", params_.joint_lower_limits);
    set_param_from_json_vector("joint_upper_limits", params_.joint_upper_limits);

    // Warn user that use_imu should be set in the robot description
    if (j.find("use_imu") != j.end()) {
      params_.use_imu = j["use_imu"];
      RCLCPP_WARN(get_node()->get_logger(),
                  "From JSON, setting params_use_imu=%d. Verify robot description has proper value "
                  "of use_imu too.",
                  params_.use_imu);
    }

    if (j.find("observation_history") != j.end()) {
      params_.observation_history = j["observation_history"];
      RCLCPP_INFO(get_node()->get_logger(), "From JSON, setting params_.observation_history=%ld",
                  params_.observation_history);
    }

    // Frame layout. mjlab-exported policies declare it as the "obs_spec"
    // component list, which the assembler below follows verbatim; policies from
    // the original lab pipeline have none and get the implicit proprio layout.
    if (j.find("obs_spec") != j.end()) {
      // Throws (and is caught below) on an unknown component or a bad size.
      obs_layout_ = parse_obs_spec(j["obs_spec"], kActionSize);
      if (j["obs_spec"].contains("history") &&
          static_cast<long>(j["obs_spec"]["history"]) != params_.observation_history) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "obs_spec.history (%ld) != observation_history (%ld)",
                     static_cast<long>(j["obs_spec"]["history"]),
                     params_.observation_history);
        return controller_interface::CallbackReturn::ERROR;
      }
      if (j.find("single_observation_size") != j.end() &&
          static_cast<int>(j["single_observation_size"]) != obs_layout_.frame_size) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "obs_spec components sum to %d but single_observation_size is %d",
                     obs_layout_.frame_size, static_cast<int>(j["single_observation_size"]));
        return controller_interface::CallbackReturn::ERROR;
      }
      RCLCPP_INFO(get_node()->get_logger(), "From obs_spec, frame size is %d",
                  obs_layout_.frame_size);
    } else {
      const int declared =
          j.find("single_observation_size") != j.end() ? int(j["single_observation_size"]) : 36;
      // Throws (and is caught below) unless this is the plain proprio frame.
      obs_layout_ = legacy_obs_layout(declared, kActionSize);
      RCLCPP_INFO(get_node()->get_logger(), "No obs_spec; plain proprio frame, size %d",
                  declared);
    }
    single_observation_size_ = obs_layout_.frame_size;

    // Gait reference tables, if this is a motion-reference ("mimic") policy. The
    // block and the frame's gait_reference component must come together;
    // otherwise the policy would be fed a layout it was not trained on.
    use_gait_reference_ = j.find("gait_reference") != j.end();
    // MixedGaitsJump policies additionally carry a "jump_slot" block: the same
    // gait reference, plus a jump table insertable on the X button. The slot
    // rides on the gait clock, so it requires the gait block.
    use_jump_slot_ = j.find("jump_slot") != j.end();
    if (use_jump_slot_ && !use_gait_reference_) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "jump_slot block requires a gait_reference block (the slot overlays "
                   "the mixed gait reference)");
      return controller_interface::CallbackReturn::ERROR;
    }
    // The reference plays through the frame's gait_reference component.
    if (use_gait_reference_ != (obs_layout_.gait_reference != -1)) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "gait_reference block %s but the frame layout %s a gait_reference "
                   "component; the policy and its reference must ship together",
                   use_gait_reference_ ? "present" : "absent",
                   obs_layout_.gait_reference != -1 ? "has" : "lacks");
      return controller_interface::CallbackReturn::ERROR;
    }
    if (use_gait_reference_) {
      // Throws (and is caught below) if the block is malformed.
      parse_gait_reference(j["gait_reference"], kActionSize, gait_);
      RCLCPP_INFO(get_node()->get_logger(),
                  "Gait reference: %d phase samples, %.3f Hz (fast %.1fx above |vx|=%.2f%s), "
                  "blend_speed=%.3f",
                  gait_.n_samples, gait_.frequency, gait_.gallop_freq_mult, gait_.gallop_speed,
                  gait_.gallop_back_table.empty() ? "" : ", direction-split",
                  gait_.blend_speed);
      if (use_jump_slot_) {
        parse_jump_slot(j["jump_slot"], kActionSize, jump_slot_);
        RCLCPP_INFO(get_node()->get_logger(),
                    "Jump slot (game mode): %d samples over %.2f s (%.2f s active), grid "
                    "%.2f s, busy %.2f s, jump on button %d, run on button %d "
                    "(caps %.2f / %.2f m/s)",
                    jump_slot_.n_samples, jump_slot_.playback_s, jump_slot_.active_s,
                    jump_slot_.grid_s, jump_slot_.busy_s, jump_slot_.trigger_button,
                    jump_slot_.run_button, jump_slot_.walk_speed_cap, jump_slot_.run_speed_cap);
      }
    }

    // Trick policies (pupper-mjlab export_trick.py) carry the reference motion
    // they track. The block and the frame's motion components must come
    // together, like the gait reference above.
    use_trick_ = j.find("motion") != j.end();
    const bool trick_layout =
        obs_layout_.motion_command != -1 || obs_layout_.motion_anchor_ori_b != -1;
    if (use_trick_ != trick_layout) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "motion block %s but the frame layout %s trick components; the "
                   "policy and its motion must ship together",
                   use_trick_ ? "present" : "absent", trick_layout ? "has" : "lacks");
      return controller_interface::CallbackReturn::ERROR;
    }
    if (use_trick_) {
      if (j.find("trick") == j.end()) {
        RCLCPP_ERROR(get_node()->get_logger(), "motion block without a trick block");
        return controller_interface::CallbackReturn::ERROR;
      }
      // Throws (and is caught below) if the block is malformed.
      parse_trick_motion(j["trick"], j["motion"], kActionSize, trick_);
      RCLCPP_INFO(get_node()->get_logger(), "Trick '%s': %d frames at %.0f fps (%.1f s)",
                  std::string(j["trick"].value("name", "")).c_str(), trick_.frames,
                  trick_.fps, trick_.duration());
    }

    // Teleop command clamp: the exporter stamps the trained command ranges (or a
    // deliberately tighter cap), and the policy should never see anything outside
    // them.
    if (j.find("command_clip") != j.end()) {
      parse_command_clip(j["command_clip"], command_clip_);
      RCLCPP_INFO(get_node()->get_logger(),
                  "Command clip: vx [%.2f, %.2f], vy [%.2f, %.2f], wz [%.2f, %.2f]",
                  command_clip_.vx_lo, command_clip_.vx_hi, command_clip_.vy_lo,
                  command_clip_.vy_hi, command_clip_.wz_lo, command_clip_.wz_hi);
    }

    // IMU-yaw heading hold: the policy trained against this closed-loop yaw
    // command profile, so the same loop must run here. Default ON for gait
    // policies (kp 1.0, validated on frozen checkpoints -- the correction needs
    // no retraining), so pre-contract JSONs get it too; the block overrides,
    // kp <= 0 disables, and it needs the IMU for an absolute yaw.
    if (use_gait_reference_) {
      heading_hold_.enabled = true;
      heading_hold_.kp = 1.0;
      heading_hold_.clip = 0.3;
      heading_hold_.yaw_threshold = 0.1;
      heading_hold_.walk_threshold = 0.05;
      heading_hold_.reset();
    }
    if (j.find("heading_hold") != j.end()) {
      parse_heading_hold(j["heading_hold"], heading_hold_);
    }
    if (heading_hold_.enabled && heading_hold_.kp <= 0.0) {
      heading_hold_.enabled = false;
    }
    if (heading_hold_.enabled && !params_.use_imu) {
      RCLCPP_WARN(get_node()->get_logger(),
                  "heading hold requires use_imu for an absolute yaw; disabling");
      heading_hold_.enabled = false;
    }
    if (heading_hold_.enabled) {
      RCLCPP_INFO(get_node()->get_logger(),
                  "Heading hold: kp=%.2f clip=%.2f yaw_threshold=%.2f walk_threshold=%.2f",
                  heading_hold_.kp, heading_hold_.clip, heading_hold_.yaw_threshold,
                  heading_hold_.walk_threshold);
    }

    // Check that the observation history is consistent with the model input shape
    if (j["in_shape"].at(1) != params_.observation_history * single_observation_size_) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "observation_history (%ld) * single_observation_size (%d) != in_shape (%d)",
                   params_.observation_history, single_observation_size_,
                   static_cast<int>(j["in_shape"].at(1)));
      return controller_interface::CallbackReturn::ERROR;
    }

  } catch (const std::exception &e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (!check_param_vector_size()) {
    return controller_interface::CallbackReturn::ERROR;
  }

  // Report any trimmed joint once kp is final (the JSON has been read by now), so
  // the effective gain of a hand-tuned joint shows up in the startup log rather
  // than staying buried in a config. Worth checking against the joint's kp_max in
  // the robot description: the hardware interface clamps to it, so a large trim
  // can quietly deliver less than it asks for.
  for (int i = 0; i < kActionSize; i++) {
    if (params_.kp_scale.at(i) != 1.0) {
      RCLCPP_WARN(get_node()->get_logger(), "kp_scale for %s is %.2f: kp %.2f -> %.2f",
                  params_.joint_names.at(i).c_str(), params_.kp_scale.at(i),
                  params_.kps.at(i) * params_.gain_multiplier,
                  params_.kps.at(i) * params_.gain_multiplier * params_.kp_scale.at(i));
    }
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn NeuralController::on_configure(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  RCLCPP_INFO(get_node()->get_logger(), "configure successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration NeuralController::command_interface_configuration()
    const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::ALL};
}

controller_interface::InterfaceConfiguration NeuralController::state_interface_configuration()
    const {
  return controller_interface::InterfaceConfiguration{
      controller_interface::interface_configuration_type::ALL};
}

controller_interface::CallbackReturn NeuralController::on_activate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  // Clear command buffers to ignore pre-activation commands
  rt_cmd_vel_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::Twist>>(nullptr);
  rt_cmd_pose_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::Pose>>(nullptr);

  // A trick starts over from its first clip, aligned to the current heading.
  trick_yaw_set_ = false;
  clip_player_.reset();
  rt_clip_request_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<std_msgs::msg::String>>(nullptr);
  last_clip_request_ = nullptr;
  if (use_trick_) {
    clip_subscriber_ = get_node()->create_subscription<std_msgs::msg::String>(
        "~/clip", rclcpp::SystemDefaultsQoS(),
        [this](const std_msgs::msg::String::SharedPtr msg) { rt_clip_request_ptr_.writeFromNonRT(msg); });
    clip_status_publisher_ =
        get_node()->create_publisher<std_msgs::msg::String>("~/clip_status", rclcpp::SystemDefaultsQoS());
    rt_clip_status_publisher_ =
        std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::String>>(clip_status_publisher_);
  }

  // Populate the command interfaces map
  RCLCPP_INFO(get_node()->get_logger(), "Populating command interfaces map");
  command_interfaces_map_.clear();
  for (auto &command_interface : command_interfaces_) {
    RCLCPP_INFO(get_node()->get_logger(), "Prefix %s. Adding command interface %s",
                command_interface.get_prefix_name().c_str(),
                command_interface.get_interface_name().c_str());
    command_interfaces_map_[command_interface.get_prefix_name()].insert_or_assign(
        command_interface.get_interface_name(), std::ref(command_interface));
  }

  // Populate the state interfaces map
  state_interfaces_map_.clear();
  for (auto &state_interface : state_interfaces_) {
    RCLCPP_INFO(get_node()->get_logger(), "Prefix %s. Adding state interface %s",
                state_interface.get_prefix_name().c_str(),
                state_interface.get_interface_name().c_str());
    state_interfaces_map_[state_interface.get_prefix_name()].insert_or_assign(
        state_interface.get_interface_name(), std::ref(state_interface));
  }

  // Store the initial joint positions
  for (int i = 0; i < kActionSize; i++) {
    init_joint_pos_.at(i) =
        state_interfaces_map_.at(params_.joint_names.at(i)).at("position").get().get_value();
  }

  // Reset estop caused by falling over
  estop_active_ = false;

  init_time_ = get_node()->now();
  repeat_action_counter_ = -1;

  cmd_x_vel_ = 0.0;
  cmd_y_vel_ = 0.0;
  cmd_yaw_vel_ = 0.0;

  // Re-arm the heading hold; the target recaptures on the next quiet edge.
  heading_hold_.reset();

  rt_joy_ptr_ = realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::Joy>>(nullptr);

  // Clear any pending jump slots; the trigger button starts "held" so a button
  // that happens to be down through the controller switch needs a release
  // before it schedules a jump. Same for the L1+R1 chord -- the press that
  // switched this controller in is still down.
  slot_starts_.fill(std::numeric_limits<double>::infinity());
  slot_hop_.fill(-1);
  slot_button_prev_ = true;
  chord_prev_ = true;

  // Initialize the observation vector
  observation_.assign(params_.observation_history * single_observation_size_, 0.0);

  // Set the gravity z-component in the initial observation vector. The gait
  // reference dims stay zero, which is what a zero velocity command produces.
  if (obs_layout_.projected_gravity != -1) {
    for (int i = 0; i < params_.observation_history; i++) {
      observation_.at(i * single_observation_size_ + obs_layout_.projected_gravity + 2) = -1.0;
    }
  }

  // A re-activation must not hold a heading captured in a previous run.
  heading_hold_.reset();

  // Initialize the command subscriber
  cmd_vel_subscriber_ = get_node()->create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", rclcpp::SystemDefaultsQoS(),
      [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
        rt_cmd_vel_ptr_.writeFromNonRT(msg);
      });

  cmd_pose_subscriber_ = get_node()->create_subscription<geometry_msgs::msg::Pose>(
      "/cmd_pose", rclcpp::SystemDefaultsQoS(),
      [this](const geometry_msgs::msg::Pose::SharedPtr msg) {
        rt_cmd_pose_ptr_.writeFromNonRT(msg);
      });

  emergency_stop_subscriber_ = get_node()->create_subscription<std_msgs::msg::Empty>(
      "/emergency_stop", rclcpp::SystemDefaultsQoS(),
      [this](const std_msgs::msg::Empty::SharedPtr /*msg*/) {
        estop_active_ = true;
        RCLCPP_INFO(get_node()->get_logger(), "Emergency stop triggered");
      });

  // Gamepad. Wired up for jump policies (R2 trigger), jump-slot policies
  // (X jump + circle run cap), and any controller with a chord partner
  // (L1+R1 switch); plain gait/velocity policies stay untouched by joystick
  // chatter. Chord detection lives here rather than in update() because the
  // controller_manager switch is a service call -- non-realtime by nature.
  if (!params_.chord_partner.empty()) {
    chord_client_ = get_node()->create_client<controller_manager_msgs::srv::SwitchController>(
        "/controller_manager/switch_controller");
    RCLCPP_INFO(get_node()->get_logger(), "L1+R1 chord switches to controller '%s'",
                params_.chord_partner.c_str());
  } else {
    chord_client_ = nullptr;
  }
  // Reference tricks: tables played once through the jump-slot overlay (see
  // update()). Only for jump-slot policies, and only when enabled per instance.
  rt_reference_trick_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<std_msgs::msg::Float32MultiArray>>(nullptr);
  last_reference_trick_ = nullptr;
  if (use_jump_slot_ && params_.reference_tricks) {
    reference_trick_subscriber_ =
        get_node()->create_subscription<std_msgs::msg::Float32MultiArray>(
            "~/reference_trick", rclcpp::SystemDefaultsQoS(),
            [this](const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
              rt_reference_trick_ptr_.writeFromNonRT(msg);
            });
    RCLCPP_INFO(get_node()->get_logger(), "Accepting reference tricks on ~/reference_trick");
  }
  if (use_jump_slot_ || chord_client_) {
    joy_subscriber_ = get_node()->create_subscription<sensor_msgs::msg::Joy>(
        "/joy", rclcpp::SystemDefaultsQoS(),
        [this](const sensor_msgs::msg::Joy::SharedPtr msg) {
          rt_joy_ptr_.writeFromNonRT(msg);
          // Local copy: on_deactivate resets chord_client_ from another thread.
          auto client = chord_client_;
          if (client == nullptr) {
            return;
          }
          const auto &b = msg->buttons;
          const bool chord = kChordButtonA < static_cast<int>(b.size()) &&
                             kChordButtonB < static_cast<int>(b.size()) &&
                             b[kChordButtonA] != 0 && b[kChordButtonB] != 0;
          if (chord && !chord_prev_) {
            auto req = std::make_shared<
                controller_manager_msgs::srv::SwitchController::Request>();
            req->activate_controllers.push_back(params_.chord_partner);
            req->deactivate_controllers.push_back(get_node()->get_name());
            req->strictness =
                controller_manager_msgs::srv::SwitchController::Request::BEST_EFFORT;
            req->activate_asap = true;
            // The empty response callback prunes the pending request; the
            // switch's effect is observable directly (this controller stops).
            client->async_send_request(
                req, [](rclcpp::Client<
                         controller_manager_msgs::srv::SwitchController>::SharedFuture) {});
            RCLCPP_INFO(get_node()->get_logger(), "L1+R1: switching %s -> %s",
                        get_node()->get_name(), params_.chord_partner.c_str());
          }
          chord_prev_ = chord;
        });
  }

  // emergency_stop_reset_subscriber_ = get_node()->create_subscription<std_msgs::msg::Empty>(
  //     "/emergency_stop_reset", rclcpp::SystemDefaultsQoS(),
  //     [this](const std_msgs::msg::Empty::SharedPtr /*msg*/) {
  //       if (estop_active_) {
  //         estop_active_ = false;
  //         on_activate(rclcpp_lifecycle::State());
  //         RCLCPP_INFO(get_node()->get_logger(), "Emergency stop released");
  //       }
  //     });

  // Initialize the publishers
  policy_output_publisher_ =
      get_node()->create_publisher<ActionMsg>("~/policy_output", rclcpp::SystemDefaultsQoS());
  rt_policy_output_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<ActionMsg>>(policy_output_publisher_);

  position_command_publisher_ =
      get_node()->create_publisher<ActionMsg>("~/position_command", rclcpp::SystemDefaultsQoS());
  rt_position_command_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<ActionMsg>>(position_command_publisher_);

  observation_publisher_ =
      get_node()->create_publisher<ObservationMsg>("~/observation", rclcpp::SystemDefaultsQoS());
  rt_observation_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<ObservationMsg>>(observation_publisher_);

  // Create IMU latency publishers
  imu_latency_publisher_ = get_node()->create_publisher<std_msgs::msg::Float32>(
      "~/imu_latency_seconds", rclcpp::SystemDefaultsQoS());
  rt_imu_latency_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float32>>(
          imu_latency_publisher_);

  // Create policy inference latency publishers
  policy_inference_latency_publisher_ = get_node()->create_publisher<std_msgs::msg::Float32>(
      "~/policy_inference_latency_seconds", rclcpp::SystemDefaultsQoS());
  rt_policy_inference_latency_publisher_ =
      std::make_shared<realtime_tools::RealtimePublisher<std_msgs::msg::Float32>>(
          policy_inference_latency_publisher_);

  RCLCPP_INFO(get_node()->get_logger(), "activate successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn NeuralController::on_error(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  return controller_interface::CallbackReturn::FAILURE;
}

controller_interface::CallbackReturn NeuralController::on_deactivate(
    const rclcpp_lifecycle::State & /*previous_state*/) {
  rt_cmd_vel_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::Twist>>(nullptr);
  rt_cmd_pose_ptr_ =
      realtime_tools::RealtimeBuffer<std::shared_ptr<geometry_msgs::msg::Pose>>(nullptr);

  for (auto &command_interface : command_interfaces_) {
    command_interface.set_value(0.0);
  }
  for (int i = 0; i < kActionSize; i++) {
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("kd")
        .get()
        .set_value(params_.estop_kd);
  }

  // Tear down the gamepad subscription and chord client: an inactive
  // controller must not keep watching L1+R1, or it would answer the chord at
  // the same time as the controller that replaced it and switch right back.
  joy_subscriber_.reset();
  chord_client_.reset();
  reference_trick_subscriber_.reset();
  clip_subscriber_.reset();

  // Clear command and state interfaces maps
  command_interfaces_map_.clear();
  state_interfaces_map_.clear();

  // Release underlying command and state interfaces
  command_interfaces_.clear();
  state_interfaces_.clear();

  // Release command and state interfaces from superclass
  release_interfaces();

  RCLCPP_INFO(get_node()->get_logger(), "Deactivate successful");
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type NeuralController::update(const rclcpp::Time &time,
                                                           const rclcpp::Duration &period) {
  // When started, return to the default joint positions
  double time_since_init = (time - init_time_).seconds();
  if (time_since_init < params_.init_duration) {
    for (int i = 0; i < kActionSize; i++) {
      // Interpolate between the initial joint positions and the default joint
      // positions
      double interpolated_joint_pos =
          init_joint_pos_.at(i) * (1 - time_since_init / params_.init_duration) +
          params_.default_joint_pos.at(i) * (time_since_init / params_.init_duration);
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("position")
          .get()
          .set_value(interpolated_joint_pos);
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("kp")
          .get()
          .set_value(params_.init_kps.at(i));
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("kd")
          .get()
          .set_value(params_.init_kds.at(i));
    }
    return controller_interface::return_type::OK;
  }

  // After the init_duration has passed, fade in the policy actions
  double time_since_fade_in = (time - init_time_).seconds() - params_.init_duration;
  float fade_in_multiplier = std::min(time_since_fade_in / params_.fade_in_duration, 1.0);

  // Only get a new action from the policy when repeat_action_counter_ is 0
  repeat_action_counter_ += 1;
  repeat_action_counter_ %= params_.repeat_action;
  if (repeat_action_counter_ != 0) {
    return controller_interface::return_type::OK;
  }

  // Get the latest commanded velocities
  auto cmd_vel = rt_cmd_vel_ptr_.readFromRT();
  if (cmd_vel && cmd_vel->get()) {
    cmd_x_vel_ = cmd_vel->get()->linear.x;
    cmd_y_vel_ = cmd_vel->get()->linear.y;
    cmd_yaw_vel_ = cmd_vel->get()->angular.z;
  }

  // Clamp the teleop command to the policy's declared bounds. The filtered
  // command -- not the raw joystick -- is what the observation and the gait
  // reference see, matching the command the policy trained on. The heading
  // hold may additionally replace cmd_yaw once the IMU is read below.
  double cmd_vx = cmd_x_vel_, cmd_vy = cmd_y_vel_, cmd_yaw_raw = cmd_yaw_vel_;
  command_clip_.apply(cmd_vx, cmd_vy, cmd_yaw_raw);
  // Per-instance speed caps on top of the policy's own bounds, e.g. the
  // walk-only slot's asymmetric forward/backward limits.
  if (params_.max_vx_forward > 0.0) {
    cmd_vx = std::min(cmd_vx, params_.max_vx_forward);
  }
  if (params_.max_vx_backward > 0.0) {
    cmd_vx = std::max(cmd_vx, -params_.max_vx_backward);
  }
  double cmd_yaw = cmd_yaw_raw;

  // Get the latest commanded pose
  auto cmd_pose = rt_cmd_pose_ptr_.readFromRT();
  if (cmd_pose && cmd_pose->get()) {
    const auto &pose_msg = *cmd_pose->get();
    tf2::Quaternion q(pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z,
                      pose_msg.orientation.w);
    desired_world_z_in_body_frame_ = tf2::Vector3(0, 0, 1);
    desired_world_z_in_body_frame_ = tf2::quatRotate(q.inverse(), desired_world_z_in_body_frame_);
  }

  // Game mode (jump-slot policies): walk-speed cap unless the run button
  // (circle) is held, and a rising edge on the trigger button (X) schedules a
  // jump at the next gait-grid point past any pending slot's busy window --
  // the same arithmetic as training's request_jump, on the fade-in clock.
  // With game_buttons false the same policy runs walk-only: buttons are
  // ignored, the walk cap always applies, and no jump is ever scheduled.
  if (use_jump_slot_) {
    bool run_held = false;
    bool trigger_pressed = false;
    auto joy = rt_joy_ptr_.readFromRT();
    if (params_.game_buttons && joy && joy->get()) {
      const auto &b = joy->get()->buttons;
      auto held = [&](int idx) {
        return idx >= 0 && idx < static_cast<int>(b.size()) && b[idx] != 0;
      };
      run_held = held(jump_slot_.run_button);
      trigger_pressed = held(jump_slot_.trigger_button);
    }
    const double cap = params_.speed_cap > 0.0
                           ? params_.speed_cap
                           : (run_held ? jump_slot_.run_speed_cap : jump_slot_.walk_speed_cap);
    // The cap applies to the filtered command (the one the observation and the
    // gait reference see), on top of the JSON's command_clip.
    cmd_vx = std::min(std::max(cmd_vx, -cap), cap);
    cmd_vy = std::min(std::max(cmd_vy, -cap), cap);

    // Retire slots whose busy window has fully passed.
    for (std::size_t i = 0; i < slot_starts_.size(); i++) {
      if (std::isfinite(slot_starts_[i]) &&
          time_since_fade_in > slot_starts_[i] + jump_slot_.busy_s) {
        slot_starts_[i] = std::numeric_limits<double>::infinity();
        slot_hop_[i] = -1;
      }
    }
    if (trigger_pressed && !slot_button_prev_ && fade_in_multiplier >= 1.0f) {
      double start =
          std::ceil((time_since_fade_in + 1e-6) / jump_slot_.grid_s) * jump_slot_.grid_s;
      for (std::size_t pass = 0; pass <= slot_starts_.size(); pass++) {
        bool blocked = false;
        for (const double s : slot_starts_) {
          if (std::isfinite(s) && s <= start && start < s + jump_slot_.busy_s) {
            start = std::ceil((s + jump_slot_.busy_s - 1e-6) / jump_slot_.grid_s) *
                    jump_slot_.grid_s;
            blocked = true;
          }
        }
        if (!blocked) {
          break;
        }
      }
      bool already = false;
      for (const double s : slot_starts_) {
        already = already || std::abs(s - start) < 1e-6;
      }
      if (!already) {
        for (std::size_t i = 0; i < slot_starts_.size(); i++) {
          auto &s = slot_starts_[i];
          if (!std::isfinite(s)) {
            s = start;
            slot_hop_[i] = -1;  // decided when the slot starts
            RCLCPP_INFO(get_node()->get_logger(), "Jump scheduled at t=%.2f s (now %.2f s)",
                        start, time_since_fade_in);
            break;
          }
        }
      }
    }

    // A reference trick: load its table into the slot and start it now. Refused
    // (and dropped) until the policy has faded in, or while a slot is pending.
    auto trick = rt_reference_trick_ptr_.readFromRT();
    if (trick && trick->get() && trick->get() != last_reference_trick_) {
      last_reference_trick_ = trick->get();
      const auto &d = trick->get()->data;
      bool busy = false;
      for (const double s : slot_starts_) {
        busy = busy || std::isfinite(s);
      }
      const int n = d.size() > 3 ? static_cast<int>((d.size() - 3) / kActionSize) : 0;
      if (n <= 0 || d.size() != 3 + static_cast<std::size_t>(n) * kActionSize) {
        RCLCPP_WARN(get_node()->get_logger(),
                    "reference trick ignored: expected 3 + n x %d values, got %zu",
                    kActionSize, d.size());
      } else if (fade_in_multiplier < 1.0f || busy) {
        RCLCPP_WARN(get_node()->get_logger(),
                    "reference trick ignored: policy %s", busy ? "busy with a slot"
                                                                : "still fading in");
      } else {
        JumpSlot next = jump_slot_;  // swap in only once it validates
        next.playback_s = d[0];
        next.active_s = d[1];
        next.cross_fade_s = d[2];
        next.n_samples = n;
        next.busy_s = std::max(next.busy_s, next.active_s);
        next.jump_table.assign(d.begin() + 3, d.end());
        if (!next.valid()) {
          RCLCPP_WARN(get_node()->get_logger(), "reference trick ignored: invalid timing");
        } else {
          jump_slot_ = std::move(next);
          slot_starts_[0] = time_since_fade_in;
          slot_hop_[0] = 0;  // a reference trick plays its own table, never the hop
          RCLCPP_INFO(get_node()->get_logger(),
                      "Reference trick: %d samples over %.2f s (%.2f s active), now",
                      n, jump_slot_.playback_s, jump_slot_.active_s);
        }
      }
    }
    slot_button_prev_ = trigger_pressed;
  }

  // If an emergency stop has been triggered, set all commands to 0, set damping, and return
  // TODO: use deactivate instead?
  if (estop_active_) {
    for (auto &command_interface : command_interfaces_) {
      command_interface.set_value(0.0);
    }
    for (int i = 0; i < kActionSize; i++) {
      command_interfaces_map_.at(params_.joint_names.at(i))
          .at("kd")
          .get()
          .set_value(params_.estop_kd);
    }
    return controller_interface::return_type::OK;
  }

  // Get the latest observation
  double ang_vel_x = 0;
  double ang_vel_y = 0;
  double ang_vel_z = 0;
  double orientation_w = 0;
  double orientation_x = 0;
  double orientation_y = 0;
  double orientation_z = 0;
  double time_since_measurement_seconds = 0;
  try {
    // read IMU states from hardware interface
    RCLCPP_DEBUG(get_node()->get_logger(), "Attempting to read IMU angular_velocity.x from %s", params_.imu_sensor_name.c_str());
    ang_vel_x = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.x")
                    .get()
                    .get_value();
    ang_vel_y = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.y")
                    .get()
                    .get_value();
    ang_vel_z = state_interfaces_map_.at(params_.imu_sensor_name)
                    .at("angular_velocity.z")
                    .get()
                    .get_value();
    orientation_w =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.w").get().get_value();
    orientation_x =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.x").get().get_value();
    orientation_y =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.y").get().get_value();
    orientation_z =
        state_interfaces_map_.at(params_.imu_sensor_name).at("orientation.z").get().get_value();

    // Try to read time_since_measurement_seconds if available (optional for simulation)
    auto imu_interfaces = state_interfaces_map_.at(params_.imu_sensor_name);
    if (imu_interfaces.find("time_since_measurement_seconds") != imu_interfaces.end()) {
      time_since_measurement_seconds = imu_interfaces.at("time_since_measurement_seconds").get().get_value();
    } else {
      // Default to 0 if not available (simulation case)
      time_since_measurement_seconds = 0.0;
      RCLCPP_DEBUG_ONCE(get_node()->get_logger(), "time_since_measurement_seconds interface not available, using default value 0.0");
    }

    // Check that the orientation is identity if we are not using the IMU. Use approximate checks
    // to avoid floating point errors
    if (!params_.use_imu) {
      if (std::abs(orientation_w - 1.0) > 1e-3 || std::abs(orientation_x) > 1e-3 ||
          std::abs(orientation_y) > 1e-3 || std::abs(orientation_z) > 1e-3) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "use_imu is false but IMU orientation is not identity");
        return controller_interface::return_type::ERROR;
      }
    } else {
      // Check that the orientation is not identity if we are using the IMU
      if (std::abs(orientation_w - 1.0) < 1e-6 && std::abs(orientation_x) < 1e-6 &&
          std::abs(orientation_y) < 1e-6 && std::abs(orientation_z) < 1e-6) {
        RCLCPP_WARN(get_node()->get_logger(),
                    "use_imu is true but IMU orientation is near identity");
      }
    }

    // Calculate the projected gravity vector
    tf2::Quaternion q(orientation_x, orientation_y, orientation_z, orientation_w);
    tf2::Matrix3x3 m(q);
    tf2::Vector3 world_gravity_vector(0, 0, -1);
    tf2::Vector3 projected_gravity_vector = m.inverse() * world_gravity_vector;

    // If the maximum body angle is exceeded, trigger an emergency stop
    if (-projected_gravity_vector[2] < cos(params_.max_body_angle)) {
      estop_active_ = true;
      RCLCPP_INFO(get_node()->get_logger(), "Emergency stop triggered");
      return controller_interface::return_type::OK;
    }

    // Heading hold: while walking with a quiet commanded yaw, the emitted yaw
    // command becomes the clipped P-correction toward the heading captured at
    // the quiet edge -- the same closed loop the policy trained against.
    cmd_yaw = heading_hold_.apply(
        yaw_from_quaternion(orientation_w, orientation_x, orientation_y, orientation_z),
        cmd_vx, cmd_vy, cmd_yaw_raw);

    // Fill the newest observation frame, at the offsets the policy's obs_spec
    // declared (-1 = the policy does not observe that component).
    if (obs_layout_.ang_vel != -1) {
      observation_.at(obs_layout_.ang_vel + 0) = (float)ang_vel_x;
      observation_.at(obs_layout_.ang_vel + 1) = (float)ang_vel_y;
      observation_.at(obs_layout_.ang_vel + 2) = (float)ang_vel_z;
    }
    if (obs_layout_.projected_gravity != -1) {
      observation_.at(obs_layout_.projected_gravity + 0) = (float)projected_gravity_vector[0];
      observation_.at(obs_layout_.projected_gravity + 1) = (float)projected_gravity_vector[1];
      observation_.at(obs_layout_.projected_gravity + 2) = (float)projected_gravity_vector[2];
    }
    if (obs_layout_.command != -1) {
      observation_.at(obs_layout_.command + 0) = (float)cmd_vx;
      observation_.at(obs_layout_.command + 1) = (float)cmd_vy;
      observation_.at(obs_layout_.command + 2) = (float)cmd_yaw;
    }
    if (obs_layout_.desired_world_z != -1) {
      observation_.at(obs_layout_.desired_world_z + 0) = (float)desired_world_z_in_body_frame_.getX();
      observation_.at(obs_layout_.desired_world_z + 1) = (float)desired_world_z_in_body_frame_.getY();
      observation_.at(obs_layout_.desired_world_z + 2) = (float)desired_world_z_in_body_frame_.getZ();
    }

    // Joint positions
    if (obs_layout_.joint_pos_rel != -1) {
      for (int i = 0; i < kActionSize; i++) {
        // Only include the joint position in the observation if the action type
        // is position
        if (params_.action_types.at(i) == "position") {
          RCLCPP_DEBUG(get_node()->get_logger(), "Attempting to read joint position for %s (index %d)", params_.joint_names.at(i).c_str(), i);
          float joint_pos =
              state_interfaces_map_.at(params_.joint_names.at(i)).at("position").get().get_value();
          observation_.at(obs_layout_.joint_pos_rel + i) =
              joint_pos - params_.default_joint_pos.at(i);
        }
      }
    }

    // Joint velocities
    if (obs_layout_.joint_vel_rel != -1) {
      for (int i = 0; i < kActionSize; i++) {
        observation_.at(obs_layout_.joint_vel_rel + i) = static_cast<float>(
            state_interfaces_map_.at(params_.joint_names.at(i)).at("velocity").get().get_value());
      }
    }

    // Trick reference: the frame playing now. The clock starts once the policy
    // has fully faded in; until then the policy holds frame 0, the pose training
    // starts every episode from.
    if (use_trick_) {
      const double trick_time = time_since_fade_in - params_.fade_in_duration;
      auto request = rt_clip_request_ptr_.readFromRT();
      if (request && request->get() && request->get() != last_clip_request_) {
        last_clip_request_ = request->get();
        if (!clip_player_.request(trick_, request->get()->data)) {
          RCLCPP_WARN(get_node()->get_logger(), "clip request '%s' refused (unknown clip or queue full)",
                      request->get()->data.c_str());
        }
      }
      const int frame = clip_player_.frame(trick_, trick_time);
      if (rt_clip_status_publisher_ && rt_clip_status_publisher_->trylock()) {
        rt_clip_status_publisher_->msg_.data =
            trick_.clips[clip_player_.clip()].name + (clip_player_.holding() ? " holding" : " playing") +
            " frame " + std::to_string(frame) + " queued " + std::to_string(clip_player_.queued());
        rt_clip_status_publisher_->unlockAndPublish();
      }
      if (obs_layout_.motion_command != -1) {
        for (int i = 0; i < kActionSize; i++) {
          observation_.at(obs_layout_.motion_command + i) =
              trick_.joint_pos.at(frame * kActionSize + i);
          observation_.at(obs_layout_.motion_command + kActionSize + i) =
              trick_.joint_vel.at(frame * kActionSize + i);
        }
      }
      if (obs_layout_.motion_anchor_ori_b != -1) {
        const Quat robot{orientation_w, orientation_x, orientation_y, orientation_z};
        const Quat ref0{trick_.anchor_quat[0], trick_.anchor_quat[1], trick_.anchor_quat[2],
                        trick_.anchor_quat[3]};
        if (!trick_yaw_set_) {
          trick_yaw_offset_ = quat_yaw(robot) - quat_yaw(ref0);
          trick_yaw_set_ = true;
        }
        const Quat ref{trick_.anchor_quat[4 * frame + 0], trick_.anchor_quat[4 * frame + 1],
                       trick_.anchor_quat[4 * frame + 2], trick_.anchor_quat[4 * frame + 3]};
        relative_orientation_6d(robot, quat_mul(yaw_quat(trick_yaw_offset_), ref),
                                observation_.data() + obs_layout_.motion_anchor_ori_b);
      }
    }

    // Reference motion the gait policy tracks. This has to be written into the
    // newest frame *before* inference (unlike the previous action, which the
    // policy-output loop below writes into the next frame after the rotate).
    if (use_gait_reference_) {
      // The phase clock is time_since_fade_in: seconds since the policy took over,
      // in double. Wall clock rather than a step count, because the controller runs
      // at 52 Hz against a policy trained at 50 Hz and it is the physical cadence of
      // the foot trajectory that has to be right.
      // The filtered (clip + heading-hold) command, matching what the
      // observation carries: training computes the reference from the emitted
      // command, so the two must see the same command here too.
      if (use_jump_slot_) {
        // The active slot, if any: the scheduled start whose active window
        // covers now. +inf means pure gait -- compute_mixed_jump_reference_offset
        // checks the window itself, but picking here keeps one slot unambiguous
        // when a second is already queued.
        // A slot latches hop-or-jump at its first active step, from the same
        // (capped) command the observation carries -- as training does.
        double slot_start = std::numeric_limits<double>::infinity();
        bool slot_is_hop = false;
        for (std::size_t i = 0; i < slot_starts_.size(); i++) {
          const double s = slot_starts_[i];
          const double t_in = time_since_fade_in - s;
          if (!std::isfinite(s) || t_in < 0.0) {
            continue;
          }
          if (slot_hop_[i] < 0) {
            slot_hop_[i] = (jump_slot_.has_hop() && cmd_vx >= jump_slot_.hop_speed) ? 1 : 0;
          }
          if (t_in < jump_slot_.window(slot_hop_[i] == 1)) {
            slot_start = s;
            slot_is_hop = slot_hop_[i] == 1;
          }
        }
        compute_mixed_jump_reference_offset(gait_, jump_slot_, params_.default_joint_pos,
                                            time_since_fade_in, cmd_vx, cmd_vy,
                                            cmd_yaw, slot_start, slot_is_hop,
                                            observation_.data() + obs_layout_.gait_reference);
      } else {
        compute_gait_reference_offset(gait_, params_.default_joint_pos, time_since_fade_in,
                                      cmd_vx, cmd_vy, cmd_yaw,
                                      observation_.data() + obs_layout_.gait_reference);
      }
    }
  } catch (const std::out_of_range &e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to read states from hardware interface - std::out_of_range exception: %s", e.what());
    
    // Check which interfaces are missing
    RCLCPP_ERROR(get_node()->get_logger(), "=== Debug Information ===");
    
    // Check IMU interface
    if (state_interfaces_map_.find(params_.imu_sensor_name) == state_interfaces_map_.end()) {
      RCLCPP_ERROR(get_node()->get_logger(), "Missing IMU sensor interface: %s", params_.imu_sensor_name.c_str());
    } else {
      RCLCPP_INFO(get_node()->get_logger(), "IMU sensor interface '%s' found", params_.imu_sensor_name.c_str());
      auto &imu_interfaces = state_interfaces_map_.at(params_.imu_sensor_name);
      std::vector<std::string> required_imu = {"angular_velocity.x", "angular_velocity.y", "angular_velocity.z", 
                                               "orientation.x", "orientation.y", "orientation.z", "orientation.w"};
      for (const auto &iface : required_imu) {
        if (imu_interfaces.find(iface) == imu_interfaces.end()) {
          RCLCPP_ERROR(get_node()->get_logger(), "Missing IMU interface: %s.%s", params_.imu_sensor_name.c_str(), iface.c_str());
        }
      }
    }
    
    // Check joint interfaces
    for (size_t i = 0; i < params_.joint_names.size(); i++) {
      const auto &joint_name = params_.joint_names.at(i);
      if (state_interfaces_map_.find(joint_name) == state_interfaces_map_.end()) {
        RCLCPP_ERROR(get_node()->get_logger(), "Missing joint interface: %s", joint_name.c_str());
      } else if (params_.action_types.at(i) == "position") {
        auto &joint_interfaces = state_interfaces_map_.at(joint_name);
        if (joint_interfaces.find("position") == joint_interfaces.end()) {
          RCLCPP_ERROR(get_node()->get_logger(), "Missing position interface for joint: %s", joint_name.c_str());
        }
      }
    }
    
    // List all available interfaces for debugging
    RCLCPP_ERROR(get_node()->get_logger(), "Available state interfaces:");
    for (const auto &[name, interfaces] : state_interfaces_map_) {
      std::string interface_list;
      for (const auto &[iface_name, iface_ref] : interfaces) {
        if (!interface_list.empty()) interface_list += ", ";
        interface_list += iface_name;
      }
      RCLCPP_ERROR(get_node()->get_logger(), "  %s: [%s]", name.c_str(), interface_list.c_str());
    }
    RCLCPP_ERROR(get_node()->get_logger(), "========================");
    
    return controller_interface::return_type::ERROR;
  }

  // Clip the observation vector
  for (auto &obs : observation_) {
    obs = std::clamp(obs, static_cast<float>(-params_.observation_limit),
                     static_cast<float>(params_.observation_limit));
  }

  // Check observation for NaNs
  if (contains_nan(observation_)) {
    RCLCPP_ERROR(get_node()->get_logger(), "observation_ contains NaN");
    return controller_interface::return_type::ERROR;
  }

  // Publish the observation
  if (rt_observation_publisher_->trylock()) {
    // TODO make a custom msg type with header
    // rt_observation_publisher_->msg_.header.stamp = time;
    rt_observation_publisher_->msg_.data = observation_;
    rt_observation_publisher_->unlockAndPublish();
  }

  // Measure the time before policy inference
  auto start_time = std::chrono::high_resolution_clock::now();

  // Perform policy inference
  model_->forward(observation_.data());

  // Measure the time after policy inference
  auto end_time = std::chrono::high_resolution_clock::now();
  auto inference_duration_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  RCLCPP_INFO(get_node()->get_logger(),
              "Policy inference took %.2f ms\tIMU measurement age: %.3f ms",
              inference_duration_us / 1000.0, time_since_measurement_seconds * 1000.0);

  // Shift the observation history to the right by one frame for the next control
  // step https://en.cppreference.com/w/cpp/algorithm/rotate
  std::rotate(observation_.rbegin(), observation_.rbegin() + single_observation_size_,
              observation_.rend());

  // Process the actions
  const float *policy_output = model_->getOutputs();

  // Publish the policy output
  if (rt_policy_output_publisher_->trylock()) {
    rt_policy_output_publisher_->msg_.data.resize(kActionSize, 0.0);
    for (int i = 0; i < kActionSize; i++) {
      rt_policy_output_publisher_->msg_.data.at(i) = policy_output[i];
    }
    // rt_policy_output_publisher_->msg_.header.stamp = time;
    rt_policy_output_publisher_->unlockAndPublish();
  }

  for (int i = 0; i < kActionSize; i++) {
    float action = policy_output[i];
    float action_scale = params_.action_scales.at(i);
    float default_joint_pos = params_.default_joint_pos.at(i);
    float lower_limit = params_.joint_lower_limits.at(i);
    float upper_limit = params_.joint_upper_limits.at(i);

    // Copy policy_output to the observation vector
    if (obs_layout_.last_action != -1) {
      observation_.at(obs_layout_.last_action + i) = fade_in_multiplier * action;
    }
    // Scale and de-normalize to get the action vector
    if (params_.action_types.at(i) == "position") {
      float unclipped = fade_in_multiplier * action * action_scale + default_joint_pos;
      action_.at(i) = std::clamp(unclipped, lower_limit, upper_limit);
    } else {
      action_.at(i) = fade_in_multiplier * action * action_scale;
    }

    if (std::isnan(action_.at(i))) {
      RCLCPP_ERROR(get_node()->get_logger(), "action_[%d] is NaN", i);
      return controller_interface::return_type::ERROR;
    }

    // Send the action to the hardware interface
    // Multiply by the gain multiplier to scale the gains to account for real2sim gap,
    // and by the per-joint kp_scale to compensate for this robot's motors
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at(params_.action_types.at(i))
        .get()
        .set_value((double)action_.at(i));
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("kp")
        .get()
        .set_value(params_.kps.at(i) * params_.gain_multiplier * params_.kp_scale.at(i));
    command_interfaces_map_.at(params_.joint_names.at(i))
        .at("kd")
        .get()
        .set_value(params_.kds.at(i) * params_.gain_multiplier);
  }

  // Publish the scaled and final position command
  if (rt_position_command_publisher_->trylock()) {
    rt_position_command_publisher_->msg_.data.resize(kActionSize, 0.0);
    for (int i = 0; i < kActionSize; i++) {
      rt_position_command_publisher_->msg_.data.at(i) = action_.at(i);
    }
    // rt_position_command_publisher_->msg_.header.stamp = time;
    rt_position_command_publisher_->unlockAndPublish();
  }

  // Publish imu latency
  if (rt_imu_latency_publisher_->trylock()) {
    rt_imu_latency_publisher_->msg_.data = time_since_measurement_seconds;
    rt_imu_latency_publisher_->unlockAndPublish();
  }

  if (rt_policy_inference_latency_publisher_->trylock()) {
    rt_policy_inference_latency_publisher_->msg_.data = inference_duration_us / 1000000.0;
    rt_policy_inference_latency_publisher_->unlockAndPublish();
  }

  // Get the policy inference time
  // double policy_inference_time = (get_node()->now() - time).seconds();
  // RCLCPP_INFO(get_node()->get_logger(), "policy inference time: %f",
  // policy_inference_time);

  return controller_interface::return_type::OK;
}

}  // namespace neural_controller

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(neural_controller::NeuralController,
                       controller_interface::ControllerInterface)