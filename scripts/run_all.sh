#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_all.sh [--no-install] [-- <extra args to detect_triple_riding.py>]
# Examples:
#   ./scripts/run_all.sh
#   ./scripts/run_all.sh --no-install -- --conf 0.5

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR=".venv"
PYTHON="$VENV_DIR/bin/python3"
PIP="$VENV_DIR/bin/pip"

NO_INSTALL=0
EXTRA_ARGS=()

# parse args: support --no-install and pass the rest after -- to the python script
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-install)
      NO_INSTALL=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "$PYTHON" ]]; then
  echo "Creating venv at $VENV_DIR (python3 -m venv)..."
  python3 -m venv "$VENV_DIR"
fi

if [[ $NO_INSTALL -ne 1 ]]; then
  echo "Upgrading pip and installing dependencies into $VENV_DIR..."
  "$PYTHON" -m pip install --upgrade pip

  if [[ -f "requirements.txt" ]]; then
    "$PIP" install -r requirements.txt
  fi

  # Ensure the core runtime deps are present
  "$PIP" install --upgrade ultralytics pillow numpy opencv-python
fi

# Run the detection script with the provided defaults; user can override flags by passing extra args
echo "Running detect_triple_riding.py with embedded defaults (you can pass extra args)."

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  # pass additional user args
  "$PYTHON" detect_triple_riding.py \
    --images datasets/triple_riding/images \
    --weights yolov8n.pt \
    --conf 0.3 --dist-thresh 0.75 \
    --compact-violation --show-confidence --show-arrow "${EXTRA_ARGS[@]}"
else
  # no extra args
  "$PYTHON" detect_triple_riding.py \
    --images datasets/triple_riding/images \
    --weights yolov8n.pt \
    --conf 0.3 --dist-thresh 0.75 \
    --compact-violation --show-confidence --show-arrow
fi

exit 0
