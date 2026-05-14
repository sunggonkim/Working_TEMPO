#!/bin/bash
# =============================================================================
# Helper: submit all 5 scaling jobs (2, 4, 8, 16, 32 nodes) at once
# Usage: bash eval/node_scaling/submit_all_scaling.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/run_node_scaling.slurm"

for N in 2 4 8 16 32; do
    JOBID=$(sbatch --nodes="${N}" --parsable "${SLURM_SCRIPT}")
    echo "Submitted N=${N}: job ${JOBID}"
done

echo ""
echo "Monitor with: squeue -u \$USER -o '%.10i %.8j %.6D %.5R %T'"
echo "After all complete, plot with:"
echo "  python eval/node_scaling/plot_node_scaling.py"
