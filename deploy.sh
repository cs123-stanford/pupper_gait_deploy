#!/usr/bin/env bash
# One command to put the current policy + controller on the robot and run it.
#
#   ./deploy.sh                    # install, rebuild, launch
#   ./deploy.sh mjlab/pdfzwf3l     # also download that run's policy first
#   ./deploy.sh --no-launch        # stop after the rebuild
#
# Anything after -- is passed straight to ros2 launch, e.g.
#   ./deploy.sh -- sim:=True
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

RUN=""
DO_LAUNCH=1
LAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-launch) DO_LAUNCH=0; shift ;;
    --) shift; LAUNCH_ARGS=("$@"); break ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) RUN="$1"; shift ;;
  esac
done

# W&B now issues ~86-character API keys; wandb < 0.22.3 rejects anything that
# isn't 40 characters, so 'wandb login' fails on the version older Pupper images
# ship. Upgrade (only when too old) before anything talks to W&B.
WANDB_MIN="0.22.3"
wandb_ok() {
  python3 - "$WANDB_MIN" <<'PY'
import re, sys
def ver(s): return tuple(int(x) for x in re.findall(r"\d+", s)[:3])
try:
    import wandb
except Exception:
    sys.exit(1)
sys.exit(0 if ver(wandb.__version__) >= ver(sys.argv[1]) else 1)
PY
}
if ! wandb_ok; then
  echo ">>> upgrading wandb to >= $WANDB_MIN (older versions reject new W&B API keys)"
  PIP=(python3 -m pip install --user --upgrade "wandb>=$WANDB_MIN")
  "${PIP[@]}" || "${PIP[@]}" --break-system-packages || true
  if wandb_ok; then
    echo ">>> wandb upgraded. Log in (once) with:  python3 -m wandb login --relogin"
  elif [[ -n "$RUN" ]]; then
    echo "!!! could not upgrade wandb (no internet?); policy download will fail" >&2
    exit 1
  else
    echo "!!! could not upgrade wandb (no internet?); continuing without it" >&2
  fi
fi

if [[ -n "$RUN" ]]; then
  echo ">>> downloading policy from $RUN"
  python3 download_policy.py "$RUN"
elif [[ -f policies/test_policy.json ]]; then
  echo ">>> validating the deployed student policy"
  python3 download_policy.py --validate-only policies/test_policy.json
fi

echo ">>> installing into the monorepo and rebuilding"
python3 install.py

if [[ $DO_LAUNCH -eq 0 ]]; then
  echo ">>> skipping launch (--no-launch)"
  exit 0
fi

echo ">>> ros2 launch neural_controller launch.py ${LAUNCH_ARGS[*]:-}"
echo ">>> press square on the joystick to activate your policy"
cd "$HOME"
exec ros2 launch neural_controller launch.py "${LAUNCH_ARGS[@]}"
