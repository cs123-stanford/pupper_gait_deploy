#!/usr/bin/env python3
"""Download an mjlab-exported policy JSON from a W&B run and validate it.

mjlab logs training checkpoints as .pt files, which the Pi cannot convert (that
needs a GPU and MuJoCo-Warp). The conversion happens on the training machine, and
Pupper runs now do it automatically: when a run ends -- on a normal finish or on
Ctrl+C -- policy.json is uploaded to the W&B run's Files. This script pulls it back
down. A run that was killed outright (SIGKILL, preemption) leaves no JSON and needs
a manual export against its last checkpoint.

To convert an older checkpoint by hand instead:

    uv run export-pupper-policy Mjlab-Trot-Bumpy-Pupper-v3 \
        --wandb-run-path mjlab/pdfzwf3l --upload-wandb

    python3 download_policy.py mjlab/pdfzwf3l        # entity/project/run-id
    python3 download_policy.py pdfzwf3l --project mjlab
    python3 download_policy.py --file policy.json --output policies/test_policy.json

Validation is not cosmetic: a frame-size or joint-order mismatch between the
policy and the controller config is silent on the robot until the legs move.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

# Joint order the controller config uses; the policy's observation and action
# layout is defined in this order, so a mismatch is unrecoverable.
EXPECTED_JOINT_NAMES = [
  "leg_front_r_1", "leg_front_r_2", "leg_front_r_3",
  "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
  "leg_back_r_1", "leg_back_r_2", "leg_back_r_3",
  "leg_back_l_1", "leg_back_l_2", "leg_back_l_3",
]  # fmt: skip

BASE_OBSERVATION_SIZE = 36
ACTION_SIZE = 12

# The controller's observation provider menu (see obs_spec.hpp): every obs_spec
# component must be one of these, at exactly this size, or the controller
# cannot produce it.
KNOWN_COMPONENTS = {
  "base_ang_vel": 3,
  "projected_gravity": 3,
  "command": 3,
  "desired_world_z": 3,
  "joint_pos_rel": ACTION_SIZE,
  "last_action": ACTION_SIZE,
  "gait_reference": ACTION_SIZE,
  "joint_vel_rel": ACTION_SIZE,
  "motion_command": 2 * ACTION_SIZE,
  "motion_anchor_ori_b": 6,
}


def resolve_run_path(run: str, project: str | None, entity: str | None) -> str:
  """Accept 'entity/project/run', 'project/run', or a bare run id."""
  import wandb

  parts = run.split("/")
  if len(parts) == 3:
    return run
  api = wandb.Api()
  if len(parts) == 2:
    project, run_id = parts
  else:
    run_id = parts[0]
    if project is None:
      raise SystemExit("Pass --project (or a full entity/project/run path).")
  entity = entity or api.default_entity
  if entity is None:
    raise SystemExit("Could not determine the W&B entity. Run 'wandb login'.")
  return f"{entity}/{project}/{run_id}"


# W&B writes its own JSON into every run's files. Downloading one of those instead
# of the policy produced a confusing KeyError deep in validation, so they are
# excluded from the fallback search by name.
WANDB_INTERNAL_PREFIXES = ("wandb-", "wandb/", "media/", "config.yaml", "requirements")


def _is_candidate_policy(name: str) -> bool:
  if not name.endswith(".json"):
    return False
  return not any(name.startswith(p) for p in WANDB_INTERNAL_PREFIXES)


def fetch(run_path: str, file_name: str, dest_dir: Path) -> Path:
  """Download ``file_name`` from the run's files, falling back to artifacts."""
  import wandb

  api = wandb.Api()
  run = api.run(run_path)
  print(f"Run: {run.name} ({run_path})")

  names = [f.name for f in run.files()]
  if file_name in names:
    candidates = [file_name]
  else:
    candidates = sorted(n for n in names if _is_candidate_policy(n))
    if candidates:
      print(f"'{file_name}' not found; falling back to: {candidates[0]}")
  if candidates:
    name = candidates[0]
    print(f"Downloading run file: {name}")
    run.file(name).download(root=str(dest_dir), replace=True)
    return dest_dir / name

  for art in run.logged_artifacts():
    if ".json" in art.name:
      print(f"Downloading artifact: {art.name}")
      root = Path(art.download(root=str(dest_dir)))
      jsons = sorted(root.glob("*.json"))
      if jsons:
        return jsons[0]

  raise SystemExit(
    f"No policy JSON in {run_path} (it has {len(names)} files, none of them a "
    f"policy).\n\n"
    "mjlab uploads policy.json when a run ends -- on a normal finish or on "
    "Ctrl+C. A run that is still training, or that was killed outright "
    "(SIGKILL, cluster preemption, a crash), never gets that far.\n\n"
    "Export it by hand from the training machine:\n"
    f"  uv run export-pupper-policy <TASK-ID> --wandb-run-path {run_path} --upload-wandb"
  )


def validate(path: Path) -> dict:
  """Check the deploy contract, and report what the controller will do with it."""
  policy = json.loads(path.read_text())

  # Fail with something readable if this is not a policy at all. Downloading the
  # wrong file used to surface as a KeyError from the middle of validation, which
  # says nothing about what actually went wrong.
  missing = [k for k in ("in_shape", "layers") if k not in policy]
  if missing:
    keys = ", ".join(sorted(policy)[:8]) or "(none)"
    raise SystemExit(
      f"{path.name} is not an mjlab policy export -- missing {', '.join(missing)}.\n"
      f"Its top-level keys are: {keys}\n\n"
      "If this is 'wandb-metadata.json' or similar, the run had no policy.json "
      "and an unrelated file was picked up. Export the policy from the training "
      "machine (see this script's docstring), or pass --file to name the right one."
    )

  history = policy.get("observation_history")
  frame = policy.get("single_observation_size", BASE_OBSERVATION_SIZE)
  in_dim = policy["in_shape"][1]
  gait = policy.get("gait_reference")
  slot = policy.get("jump_slot")
  spec = policy.get("obs_spec")

  problems = []

  # The frame layout is declared by the policy, and the controller follows it.
  has_gait_component = False
  if spec is None:
    problems.append(
      "missing obs_spec: this JSON was made by an old exporter -- re-export the "
      "policy with current mjlab"
    )
  else:
    if spec.get("frame_order") != "newest_first":
      problems.append(
        f"obs_spec.frame_order is {spec.get('frame_order')!r}; the controller "
        "only implements newest_first"
      )
    seen = []
    spec_size = 0
    for component in spec.get("components", []):
      name, size = component.get("name"), component.get("size")
      if name not in KNOWN_COMPONENTS:
        problems.append(
          f"obs_spec component {name!r} is not in the controller's provider menu"
        )
      elif size != KNOWN_COMPONENTS[name]:
        problems.append(
          f"obs_spec component {name} has size {size}, expected {KNOWN_COMPONENTS[name]}"
        )
      if name in seen:
        problems.append(f"obs_spec lists {name} twice")
      seen.append(name)
      spec_size += size or 0
    if spec_size != frame:
      problems.append(
        f"obs_spec components sum to {spec_size} but single_observation_size is {frame}"
      )
    if spec.get("history") not in (None, history):
      problems.append(
        f"obs_spec.history ({spec.get('history')}) != observation_history ({history})"
      )
    has_gait_component = "gait_reference" in seen

  if history is None:
    problems.append("missing observation_history")
  elif history * frame != in_dim:
    problems.append(
      f"observation_history ({history}) * single_observation_size ({frame}) != in_shape ({in_dim})"
    )
  for key in ("default_joint_pos", "joint_lower_limits", "joint_upper_limits"):
    if key not in policy:
      problems.append(f"missing {key}")
    elif len(policy[key]) != ACTION_SIZE:
      problems.append(f"{key} has {len(policy[key])} entries, expected {ACTION_SIZE}")
  if policy["layers"][-1]["shape"][1] != ACTION_SIZE:
    problems.append(f"policy outputs {policy['layers'][-1]['shape'][1]} actions, expected {ACTION_SIZE}")

  # The gait block and the frame's gait_reference dims must come together: a
  # reference the policy never observes is dead weight, and reference dims the
  # controller cannot fill would feed the policy an input it has never seen.
  if gait is not None and not has_gait_component:
    problems.append(
      "gait_reference block present but the frame layout has no gait_reference component"
    )
  if gait is None and has_gait_component:
    problems.append(
      "frame layout has a gait_reference component but there is no gait_reference block"
    )

  if gait is not None:
    n = gait.get("n_samples", 0)
    # lift_table / fast_back_table are optional: mixed-gait policies ship them
    # (lift-in-place while turning, a separate backward-fast capture); older
    # trot/gallop policies don't.
    tables = ["trot_table", "gallop_table"]
    for optional in ("lift_table", "gallop_back_table"):
      if optional in gait:
        tables.append(optional)
    for table in tables:
      rows = gait.get(table, [])
      if len(rows) != n or any(len(r) != ACTION_SIZE for r in rows):
        problems.append(f"{table} is not {n}x{ACTION_SIZE}")
    if gait.get("joint_names") and gait["joint_names"] != EXPECTED_JOINT_NAMES:
      problems.append("gait_reference joint_names do not match the controller config order")

  # The game-mode jump slot rides on the gait reference; it never stands alone.
  if slot is not None:
    if gait is None:
      problems.append("jump_slot present without a gait_reference block")
    n = slot.get("n_samples", 0)
    rows = slot.get("jump_table", [])
    if len(rows) != n or any(len(r) != ACTION_SIZE for r in rows):
      problems.append(f"jump_slot table is not {n}x{ACTION_SIZE}")
    for key in ("playback_s", "active_s", "cross_fade_s", "grid_s", "busy_s"):
      if key not in slot:
        problems.append(f"jump_slot missing {key}")

  # Trick (motion-tracking) policies: the motion block and the frame's motion
  # components must come together, and every frame must be complete.
  motion = policy.get("motion")
  has_motion_component = spec is not None and any(
    c["name"] in ("motion_command", "motion_anchor_ori_b") for c in spec["components"]
  )
  if (motion is not None) != has_motion_component:
    problems.append(
      "motion block and the frame's motion components must ship together "
      f"(block {'present' if motion is not None else 'absent'})"
    )
  if motion is not None:
    if "trick" not in policy or "fps" not in policy["trick"]:
      problems.append("motion block without a trick block (name, fps)")
    frames = len(motion.get("joint_pos", []))
    for key, width in (("joint_pos", ACTION_SIZE), ("joint_vel", ACTION_SIZE), ("anchor_quat_wxyz", 4)):
      rows = motion.get(key, [])
      if len(rows) != frames or any(len(r) != width for r in rows):
        problems.append(f"motion.{key} is not {frames}x{width}")
    clips = policy.get("trick", {}).get("clips", [])
    names = [c.get("name") for c in clips]
    for c in clips:
      start, end = c.get("start", -1), c.get("end", -1)
      if not (0 <= start and end <= frames and end - start >= 2):
        problems.append(f"trick clip {c.get('name')!r} has a bad frame range [{start}, {end})")
      if c.get("next") not in (None, *names):
        problems.append(f"trick clip {c.get('name')!r} continues to unknown clip {c.get('next')!r}")
    if len(set(names)) != len(names):
      problems.append("trick clip names are not unique")

  clip = policy.get("command_clip")
  if clip is not None:
    for axis, pair in clip.items():
      if axis not in ("vx", "vy", "wz"):
        problems.append(f"command_clip has unknown axis {axis!r}")
      elif len(pair) != 2 or pair[0] > pair[1]:
        problems.append(f"command_clip.{axis} is not a (lo, hi) pair: {pair}")

  print()
  if policy.get("motion") is not None:
    kind = "(trick: motion tracking)"
  elif slot:
    kind = "(mixed gaits + jump slot)"
  elif gait:
    kind = "(motion reference)"
  else:
    kind = "(plain proprio)"
  print(f"  frame size:          {frame} {kind}")
  print(f"  observation history: {history}  -> input {in_dim}")
  print(f"  kp / kd:             {policy.get('kp')} / {policy.get('kd')}")
  print(f"  action scale:        {policy.get('action_scale')}")
  if spec is not None:
    print(f"  obs layout (v2):     {' + '.join(c['name'] for c in spec['components'])}")
  if gait:
    print(f"  gait tables:         {gait['n_samples']} phase samples, {gait['frequency']:.3f} Hz")
    if "lift_table" in gait:
      split = "direction-split fast tables" if "gallop_back_table" in gait else "one fast table (time-reversed backward)"
      print(
        f"  mixed gaits:         trot -> fast above |vx| = {gait['gallop_speed']}, "
        f"lift-in-place when turning, {split}"
      )
    else:
      print(
        f"  gallop:              {gait['gallop_freq_mult']}x cadence above |vx| = {gait['gallop_speed']}"
      )
    print(f"  blend speed:         {gait['blend_speed']}")
  if slot:
    print(
      f"  jump slot:           {slot['n_samples']} samples over {slot['playback_s']:.2f} s "
      f"({slot['active_s']:.2f} s active), grid {slot['grid_s']:.2f} s, busy {slot['busy_s']:.2f} s"
    )
    print(
      f"  game buttons:        jump on {slot.get('trigger_button', 0)} (X), run on "
      f"{slot.get('run_button', 1)} (circle); caps {slot.get('walk_speed_cap', 0.49)} / "
      f"{slot.get('run_speed_cap', 1.5)} m/s"
    )
  trick = policy.get("trick")
  if trick is not None:
    print(
      f"  trick:               {trick.get('name', '?')!r}, {trick.get('frames')} frames at "
      f"{trick.get('fps')} fps ({trick.get('duration_s', 0):.1f} s)"
    )
    for c in trick.get("clips", []):
      how = "loops" if c.get("loop") else f"then {c['next']}" if c.get("next") else "then holds"
      print(f"    clip {c['name']:14} frames {c['start']}-{c['end']}, {how}")
  clip = policy.get("command_clip")
  if clip is not None:
    bounds = ", ".join(f"{axis} [{lo}, {hi}]" for axis, (lo, hi) in sorted(clip.items()))
    print(f"  command clip:        {bounds}")
  hold = policy.get("heading_hold")
  if hold is not None:
    print(
      f"  heading hold:        kp={hold['kp']} clip={hold['clip']} "
      f"(yaw quiet < {hold['yaw_threshold']}, walking > {hold['walk_threshold']})"
    )

  if problems:
    print("\nPolicy FAILED validation:")
    for p in problems:
      print(f"  - {p}")
    raise SystemExit(1)
  print("\nPolicy looks consistent with the controller config.")
  return policy


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("run", nargs="?", help="W&B run: entity/project/run-id, project/run-id, or run-id")
  parser.add_argument("--project", default="mjlab")
  parser.add_argument("--entity", default=None)
  parser.add_argument("--file", default="policy.json", help="Run file to download")
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Destination path. Default: policies/test_policy.json (the "
    "square-button slot); a policy carrying a jump_slot block routes to "
    "policies/default_policy.json (the X-button slot) instead.",
  )
  parser.add_argument("--validate-only", type=Path, default=None, help="Validate a local JSON and exit")
  args = parser.parse_args()

  if args.validate_only is not None:
    validate(args.validate_only)
    return 0
  if args.run is None:
    parser.error("a run is required (or use --validate-only)")

  run_path = resolve_run_path(args.run, args.project, args.entity)
  downloads = REPO / ".downloads"
  downloads.mkdir(exist_ok=True)
  downloaded = fetch(run_path, args.file, downloads)
  policy = validate(downloaded)

  output = args.output
  if output is None:
    # Route by kind so the default policy's slot cannot be clobbered by a
    # student deploy (and vice versa).
    if "jump_slot" in policy:
      output, kind = REPO / "policies" / "default_policy.json", "default (X slot)"
    else:
      output, kind = REPO / "policies" / "test_policy.json", "student (square slot)"
    print(f"\nPolicy kind: {kind} -> {output.name}")

  shutil.copy2(downloaded, output)
  print(f"\nWrote {output}")
  print("Next: python3 install.py")
  return 0


if __name__ == "__main__":
  sys.exit(main())
