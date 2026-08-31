#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ABR_ROOT"

python analysis/run_abr_server2_r_seed_uniform_pipeline.py --resume "$@"
