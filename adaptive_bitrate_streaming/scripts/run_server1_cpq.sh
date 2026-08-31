#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABR_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ABR_ROOT"

python analysis/run_abr_server1_cpq_pipeline.py --resume "$@"
