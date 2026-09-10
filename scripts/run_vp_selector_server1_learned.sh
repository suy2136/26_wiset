#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Stage 1: FOV-label BCE, stage 2: frozen-NBS prediction fine-tuning,
# stage 3: full test evaluation. All artifacts live below one new run root.
if [[ "${1:-run}" == "run" ]]; then
  test -f viewport_prediction/data/viewports/Jin2022/video1/5Hz/simple_5Hz_user50.csv
  test -d viewport_prediction/data/images/Jin2022_images
  PYTHONPATH="$PWD" python -m unittest \
    tests.test_frozen_patch_selector_training \
    tests.test_kinematic_patch_selector
fi
exec bash scripts/run_nbs_v19_patch_selector_two_stage.sh "${1:-run}"
