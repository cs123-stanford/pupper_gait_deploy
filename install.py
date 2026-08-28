#!/usr/bin/env python3
"""Install this repo's controller changes into the pupperv3-monorepo and rebuild.

Same idea as lab 5's rebuild_neural_controller.py, but it also ships the C++ that
teaches the neural controller to read each policy's own JSON -- the observation
frame layout, gains and any reference tables come out of the policy file, so
policies from the original lab pipeline keep loading unchanged.

    python3 install.py --dry-run     # show what would happen
    python3 install.py               # install + rebuild
    python3 install.py --no-build    # install only
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_MONOREPO = Path("/home/pi/pupperv3-monorepo")
# Per-robot motor trims, kept out of git (see robot_trims.example.yaml).
ROBOT_TRIMS = REPO / "robot_trims.yaml"


def file_mappings(monorepo: Path) -> list[tuple[Path, Path, str]]:
  """(source, destination, description), in install order."""
  pkg = monorepo / "ros2_ws" / "src" / "neural_controller"
  launch = pkg / "launch"
  return [
    (
      REPO / "src" / "gait_reference.hpp",
      pkg / "include" / "neural_controller" / "gait_reference.hpp",
      "Gait reference lookup (new header)",
    ),
    (
      REPO / "src" / "gait_reference_json.hpp",
      pkg / "include" / "neural_controller" / "gait_reference_json.hpp",
      "Gait reference JSON parser (new header, shared with the test)",
    ),
    (
      REPO / "src" / "obs_spec.hpp",
      pkg / "include" / "neural_controller" / "obs_spec.hpp",
      "Observation layout resolver (contract-v2 obs_spec + legacy layouts)",
    ),
    (
      REPO / "src" / "command_filter.hpp",
      pkg / "include" / "neural_controller" / "command_filter.hpp",
      "Command clip + IMU-yaw heading hold (from the policy JSON)",
    ),
    (
      REPO / "src" / "trick_motion.hpp",
      pkg / "include" / "neural_controller" / "trick_motion.hpp",
      "Trick reference motion + relative-orientation math (new header)",
    ),
    (
      REPO / "src" / "neural_controller.hpp",
      pkg / "include" / "neural_controller" / "neural_controller.hpp",
      "Neural controller header (runtime frame layout + gait state)",
    ),
    (
      REPO / "src" / "neural_controller.cpp",
      pkg / "src" / "neural_controller.cpp",
      "Neural controller source (parses gait_reference, fills the frame)",
    ),
    (
      REPO / "src" / "neural_controller_parameters.yaml",
      pkg / "src" / "neural_controller_parameters.yaml",
      "Controller parameter declarations (adds kp_scale)",
    ),
    (
      REPO / "src" / "CMakeLists.txt",
      pkg / "CMakeLists.txt",
      "Package CMakeLists (adds the gait parity test)",
    ),
    (
      REPO / "test" / "test_gait_reference.cpp",
      pkg / "test" / "test_gait_reference.cpp",
      "Gait reference parity test",
    ),
    (
      REPO / "test" / "test_command_filter.cpp",
      pkg / "test" / "test_command_filter.cpp",
      "Command filter / heading hold / obs_spec test",
    ),
    (
      REPO / "test" / "gait_golden.json",
      pkg / "test" / "gait_golden.json",
      "Gait parity golden data (generated from the trained env)",
    ),
    (
      REPO / "test" / "mixed_jump_golden.json",
      pkg / "test" / "mixed_jump_golden.json",
      "MixedGaitsJump parity golden data (gait + jump-slot composite cases)",
    ),
    (
      REPO / "config.yaml",
      launch / "config.yaml",
      "Controller config (the button map and controller instances)",
    ),
    (
      REPO / "launch.py",
      launch / "launch.py",
      "Launch file (spawns the controllers and broadcasters)",
    ),
    (
      REPO / "policies" / "default_policy.json",
      launch / "default_policy.json",
      "The default policy loaded by neural_controller_default (X)",
    ),
    (
      REPO / "policies" / "legacy_walk_policy.json",
      launch / "legacy_walk_policy.json",
      "The original walking policy loaded by neural_controller (triangle)",
    ),
    (
      REPO / "policies" / "three_legged_policy.json",
      launch / "three_legged_policy.json",
      "The three-legged policy loaded by neural_controller_three_legged (circle)",
    ),
    (
      REPO / "policies" / "test_policy.json",
      launch / "test_policy.json",
      "The deployed student policy loaded by neural_controller_student (square)",
    ),
  ]


def install(monorepo: Path, dry_run: bool) -> bool:
  print("=" * 70)
  print("Installing the Pupper policy controllers")
  print("=" * 70)
  if dry_run:
    print("\nDRY RUN - nothing will be written\n")

  errors = 0
  for source, destination, description in file_mappings(monorepo):
    print(f"\n{description}")
    print(f"  source:      {source}")
    print(f"  destination: {destination}")

    if not source.exists():
      if source.name in (
        "default_policy.json",
        "test_policy.json",
        "mixed_jump_golden.json",
      ):
        print("  SKIPPED: no policy here yet -- run download_policy.py first")
        continue
      print("  ERROR: source file does not exist")
      errors += 1
      continue

    if dry_run:
      print("  would replace existing file" if destination.exists() else "  would create file")
      if source.name == "config.yaml" and ROBOT_TRIMS.exists():
        print(f"  would apply this robot's kp_scale trims from {ROBOT_TRIMS.name}")
      continue

    try:
      destination.parent.mkdir(parents=True, exist_ok=True)
      if destination.exists():
        backup = destination.with_suffix(destination.suffix + ".backup")
        shutil.copy2(destination, backup)
        print(f"  backup: {backup.name}")
      shutil.copy2(source, destination)
      if source.name == "config.yaml" and ROBOT_TRIMS.exists():
        apply_robot_trims(destination, ROBOT_TRIMS)
      print("  installed")
      link_into_share(monorepo, destination)
    except OSError as e:
      print(f"  ERROR: {e}")
      errors += 1

  print(f"\n{'=' * 70}\nErrors: {errors}\n{'=' * 70}")
  return errors == 0


def apply_robot_trims(config: Path, trims_file: Path) -> None:
  """Write this robot's per-joint kp_scale trims into an installed config.yaml.

  The repo's config.yaml keeps every kp_scale at 1.0 so no robot's trims ship to
  anyone else; robot_trims.yaml (gitignored) maps a controller section name, or
  ``all`` for every section, to its 12 scales. Only the installed copy is
  edited, as text, so the config's comments survive.
  """
  import yaml

  trims = (yaml.safe_load(trims_file.read_text()) or {}).get("kp_scale") or {}
  text = config.read_text()
  # Top-level sections, as (name, start, end) spans of the text.
  heads = list(re.finditer(r"^(\w+):\n", text, flags=re.M))
  spans = [
    (m.group(1), m.start(), heads[i + 1].start() if i + 1 < len(heads) else len(text))
    for i, m in enumerate(heads)
  ]
  unknown = set(trims) - {name for name, _, _ in spans} - {"all"}
  if unknown:
    raise ValueError(f"{trims_file.name}: no config.yaml section {sorted(unknown)}")
  for name, start, end in reversed(spans):  # Back to front keeps spans valid.
    scales = trims.get(name, trims.get("all"))
    block = text[start:end]
    if scales is None or "kp_scale:" not in block:
      continue
    if len(scales) != 12 or any(float(x) < 0.0 for x in scales):
      raise ValueError(f"{trims_file.name}: {name} needs 12 non-negative scales")
    block, n = re.subn(
      r"kp_scale:\s*\[[^\]]*\]",
      "kp_scale: [" + ", ".join(f"{float(x):g}" for x in scales) + "]",
      block,
    )
    if n != 1:
      raise ValueError(f"config.yaml: {name} has {n} kp_scale entries, expected 1")
    text = text[:start] + block + text[end:]
    print(f"  {name}: kp_scale trimmed from {trims_file.name}")
  config.write_text(text)


def link_into_share(monorepo: Path, src_file: Path) -> None:
  """Symlink a launch-dir file into the installed share if it is missing there.

  The workspace builds with ``--symlink-install``, which links each launch file
  into the install share individually -- at build time. A file whose *name* is
  new since the last build (a freshly downloaded test_policy.json, a renamed
  policy) has no link yet, and the controllers resolve their model_path through
  the share, so without this they open a nonexistent file and die with a JSON
  parse error at line 1. Existing links need nothing: they point at the src
  file just copied, so new content flows through.
  """
  launch = monorepo / "ros2_ws" / "src" / "neural_controller" / "launch"
  if src_file.parent != launch:
    return
  share = (
    monorepo
    / "ros2_ws"
    / "install"
    / "neural_controller"
    / "share"
    / "neural_controller"
    / "launch"
  )
  if not share.is_dir():
    return  # Not built yet; the upcoming build will make the links itself.
  link = share / src_file.name
  if link.exists() or link.is_symlink():
    return
  link.symlink_to(src_file)
  print(f"  linked into the install share: {link}")


def rebuild(monorepo: Path, dry_run: bool) -> bool:
  ros2_ws = monorepo / "ros2_ws"
  build_script = ros2_ws / "build.sh"
  print("\n" + "=" * 70)
  print("Rebuilding the ROS 2 workspace")
  print("=" * 70)

  if not build_script.exists():
    print(f"ERROR: build script not found: {build_script}")
    return False
  if dry_run:
    print(f"would run: bash {build_script} (in {ros2_ws})")
    return True

  remove_stale_legacy_header(ros2_ws)
  started = time.time()
  result = subprocess.run(["bash", str(build_script)], cwd=str(ros2_ws), check=False)
  if result.returncode != 0:
    print(f"\nBuild FAILED (exit code {result.returncode})")
    return False
  # The monorepo's build.sh ends by sourcing local_setup.bash, so its exit code is
  # that of the source, not of colcon: a failed compile still returns 0 and the
  # robot silently launches the previous libneural_controller.so. Read colcon's
  # own per-package result instead.
  failed = failed_packages(ros2_ws / "log" / "latest_build" / "events.log", started)
  if failed is None:
    print("\nWARNING: no colcon events log from this build; trusting build.sh's exit code")
  elif failed:
    print(f"\nBuild FAILED for: {', '.join(failed)}")
    for pkg in failed:
      print(f"  see {ros2_ws / 'log' / 'latest_build' / pkg / 'stdout_stderr.log'}")
    return False
  print("\nBuild succeeded.")
  return True


def remove_stale_legacy_header(ros2_ws: Path) -> None:
  """Delete the generated top-level parameters header so it is rebuilt from scratch.

  Newer generate_parameter_library releases build this deprecated header by
  concatenating the previous file with the new one, so every parameters.yaml
  change appends another copy. Our controller includes the namespaced header when
  it exists, but deleting this one keeps it valid for anything else. Both old and
  new releases regenerate it on the next build, since it is a custom-command output.
  """
  legacy = (
    ros2_ws / "build" / "neural_controller" / "include" / "neural_controller_parameters.hpp"
  )
  if legacy.exists():
    legacy.unlink()
    print(f"removed generated header so it is rebuilt clean: {legacy}")


def failed_packages(events_log: Path, since: float = 0.0) -> list[str] | None:
  """Packages whose JobEnded rc was nonzero in colcon's events log.

  None when there is no log written at or after ``since`` to judge by.
  """
  try:
    if events_log.stat().st_mtime < since:
      return None
    text = events_log.read_text(errors="replace")
  except OSError:
    return None
  ended = re.findall(r"\((\S+)\) JobEnded: \{.*?'rc': (-?\d+)", text)
  if not ended:
    return None
  return sorted({pkg for pkg, rc in ended if rc != "0"})


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--monorepo", type=Path, default=DEFAULT_MONOREPO)
  parser.add_argument("--dry-run", "-n", action="store_true")
  parser.add_argument("--no-build", action="store_true")
  args = parser.parse_args()

  if not args.monorepo.exists():
    print(f"ERROR: monorepo not found at {args.monorepo} (pass --monorepo)")
    return 1

  if not install(args.monorepo, args.dry_run):
    print("\nInstall failed. Not rebuilding.")
    return 1

  if args.no_build:
    print("\nSkipping rebuild (--no-build).")
    return 0
  if not rebuild(args.monorepo, args.dry_run):
    return 1

  print("\nDone. Launch with: ros2 launch neural_controller launch.py")
  return 0


if __name__ == "__main__":
  sys.exit(main())
