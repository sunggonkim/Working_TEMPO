#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
SCRIPT="${ROOT}/eval/sota_4node/run_g2_fabric_raw_2node.slurm"
grep -q '^#SBATCH --nodes=2$' "${SCRIPT}"
grep -q '^#SBATCH --ntasks-per-node=4$' "${SCRIPT}"
grep -q '^#SBATCH --mail-type=NONE$' "${SCRIPT}"
grep -q 'TEMPO_RD_APPROVE_G2_RAW' "${SCRIPT}"
grep -q 'TEMPO_RD_G2_RAW_EXPECTED_SOURCE_BUNDLE_SHA256' "${SCRIPT}"
grep -q 'promotion_eligible.*false' "${SCRIPT}"
! grep -qE 'sbatch|salloc|qsub' "${SCRIPT}"
bash -n "${SCRIPT}"
echo G2_FABRIC_RAW_STATIC_PASS
