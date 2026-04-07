#!/bin/bash
# Sample fusion-born alpha particle initial conditions.

set -euo pipefail

cd "$(dirname "$0")" # always run from the folder where script is
OUTPUTS="$(pwd)/outputs"
mkdir -p "$OUTPUTS"

# ── Symlink inputs ────────────────────────────────────────────────────────────
# The script looks for boozmn.nc in CWD.
ln -sf "$(pwd)/inputs/boozmn.nc" boozmn.nc

# ── Run ───────────────────────────────────────────────────────────────────────
conda run --no-capture-output -n thea \
    python scripts/1_sample_fusion_distribution.py \
    2>&1 | tee "$OUTPUTS/log.txt"

# ── Collect outputs (runs only if the script above succeeded) ─────────────────
mv scripts/outputs/* "$OUTPUTS/"
rmdir scripts/outputs
mv stdout_*.txt "$OUTPUTS/" 2>/dev/null || true   # setup_logging writes this to CWD
rm boozmn.nc

echo "Done. Results saved to $OUTPUTS"
