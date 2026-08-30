#!/usr/bin/env bash
set -euo pipefail

[[ "${TEMPO_GO_C7_QUALIFICATION_APPROVED:-}" == YES ]] || exit 2
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || exit 2
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_JOB_NODES:-}}" == 4 ]] || exit 2
[[ -z "${SHIFTER_RUNTIME:-}${SHIFTER_IMAGE:-}${UDI:-}${CRAY_ROOTFS:-}${SLURM_CONTAINER:-}" ]] || exit 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
CONTRACT="${REPO_ROOT}/eval/sota_4node/tempo_go_c7_lmcache_victim_contract_v1.json"
[[ -s "${CONTRACT}" ]]
[[ "$(jq -er '.schema' "${CONTRACT}")" == tempo-go-c7-lmcache-victim-abba-contract-v1 ]]

while IFS=$'\t' read -r relative expected; do
  [[ "$(sha256sum "${REPO_ROOT}/${relative}" | awk '{print $1}')" == "${expected}" ]]
done < <(jq -er '.source_binding | [.component_python, .component_wrapper, .lmcache_nixl_channel][] | [.path, .sha256] | @tsv' "${CONTRACT}")
[[ "$(git -C "${REPO_ROOT}/third_party/lmcache" rev-parse HEAD)" == "$(jq -er '.source_binding.lmcache_commit' "${CONTRACT}")" ]]

RESULT_ROOT="${TEMPO_GO_C7_RESULT_ROOT:-${REPO_ROOT}/results/tempo_go_c7_lmcache_victim_job_${SLURM_JOB_ID}}"
case "${RESULT_ROOT}/" in "${REPO_ROOT}/results/"*) ;; *) exit 2 ;; esac
[[ ! -e "${RESULT_ROOT}" ]]
mkdir -p -- "${RESULT_ROOT}"

MINIMUM_ACTIVE_DURATION_S=$(jq -er '.lmcache_victim_abba.minimum_active_duration_s' "${CONTRACT}")
BLOCKS=$(jq -er '.lmcache_victim_abba.parameters.minimum_blocks' "${CONTRACT}")
MAXIMUM_BLOCKS=$(jq -er '.lmcache_victim_abba.parameters.maximum_blocks' "${CONTRACT}")
REQUESTS=$(jq -er '.lmcache_victim_abba.parameters.requests' "${CONTRACT}")
KV_MIB=$(jq -er '.lmcache_victim_abba.parameters.kv_mib' "${CONTRACT}")
FOREGROUND_MIB=$(jq -er '.lmcache_victim_abba.parameters.foreground_mib' "${CONTRACT}")
BLOCK_DELAY_S=$(jq -er '.lmcache_victim_abba.parameters.block_delay_s' "${CONTRACT}")
NCCL_TIMEOUT_S=$(jq -er '.lmcache_victim_abba.parameters.process_group_timeout_s' "${CONTRACT}")
NIXL_TIMEOUT_S=$(jq -er '.lmcache_victim_abba.parameters.nixl_transfer_timeout_s' "${CONTRACT}")
TRAFFIC_PATTERN=$(jq -er '.lmcache_victim_abba.topology.traffic_pattern' "${CONTRACT}")
[[ "${TRAFFIC_PATTERN}" == paired_1to1 ]]

mapfile -t ARM_NAMES < <(jq -er '.lmcache_victim_abba.arms[].name' "${CONTRACT}")
mapfile -t ARM_LOADS < <(jq -er '.lmcache_victim_abba.arms[].nccl_load' "${CONTRACT}")
mapfile -t ARM_TOKEN_ITERS < <(jq -er '.lmcache_victim_abba.arms[].token_iters' "${CONTRACT}")
[[ ${#ARM_NAMES[@]} -eq 4 && ${#ARM_LOADS[@]} -eq 4 && ${#ARM_TOKEN_ITERS[@]} -eq 4 ]]
[[ "${ARM_LOADS[*]}" == "control hot hot control" ]]

for index in 0 1 2 3; do
  arm_name="${ARM_NAMES[index]}"
  token_iters="${ARM_TOKEN_ITERS[index]}"
  TEMPO_GO_CROSS_LAYER_COMPONENT_APPROVED=YES \
  TEMPO_GO_CROSS_LAYER_RESULT_DIR="${RESULT_ROOT}/${arm_name}" \
  TEMPO_GO_CROSS_LAYER_SRUN_STEP_NAME="c7-lmcache-victim-${index}-${SLURM_JOB_ID}" \
  TEMPO_GO_CROSS_LAYER_EPOCH="slurm-${SLURM_JOB_ID}-c7-lmcache-victim-${index}" \
  TEMPO_GO_NCCL_COMMUNICATOR_ID="c7-lmcache-victim-${index}" \
  TEMPO_GO_CROSS_LAYER_MASTER_PORT="$((36000 + SLURM_JOB_ID % 200 + index * 256))" \
  TEMPO_GO_CROSS_LAYER_NIXL_PORT_BASE="$((38000 + SLURM_JOB_ID % 100 + index * 8))" \
  TEMPO_GO_NCCL_RAS_ADDR="127.0.0.1:$((30000 + SLURM_JOB_ID % 500 + index))" \
  TEMPO_GO_CROSS_LAYER_NO_BACKGROUND_TRANSFER=0 \
  TEMPO_GO_CROSS_LAYER_TRAFFIC_PATTERN="${TRAFFIC_PATTERN}" \
  TEMPO_GO_CROSS_LAYER_BLOCKS="${BLOCKS}" \
  TEMPO_GO_CROSS_LAYER_MAXIMUM_BLOCKS="${MAXIMUM_BLOCKS}" \
  TEMPO_GO_CROSS_LAYER_MINIMUM_ACTIVE_DURATION_S="${MINIMUM_ACTIVE_DURATION_S}" \
  TEMPO_GO_CROSS_LAYER_REQUESTS="${REQUESTS}" \
  TEMPO_GO_CROSS_LAYER_KV_MIB="${KV_MIB}" \
  TEMPO_GO_CROSS_LAYER_TOKEN_ITERS="${token_iters}" \
  TEMPO_GO_CROSS_LAYER_FOREGROUND_MIB="${FOREGROUND_MIB}" \
  TEMPO_GO_CROSS_LAYER_BLOCK_DELAY_S="${BLOCK_DELAY_S}" \
  TEMPO_GO_NCCL_TIMEOUT_SECONDS="${NCCL_TIMEOUT_S}" \
  TEMPO_GO_NIXL_TIMEOUT_SECONDS="${NIXL_TIMEOUT_S}" \
  TEMPO_GO_CROSS_LAYER_TIMEOUT_SECONDS=900 \
  TEMPO_GO_CROSS_LAYER_TIME_LIMIT=00:15:00 \
    bash "${SCRIPT_DIR}/run_lmcache_nixl_contention_2node_in_allocation.sh"
  if (( index < 3 )); then
    sleep 5
  fi
done

CONTRACT_SHA256=$(sha256sum "${CONTRACT}" | awk '{print $1}')
RESULT_ROOT_ENV="${RESULT_ROOT}" \
CONTRACT_ENV="${CONTRACT}" \
CONTRACT_SHA256_ENV="${CONTRACT_SHA256}" \
RUNNER_ENV="${BASH_SOURCE[0]}" \
ANALYZER_ENV="${REPO_ROOT}/eval/sota_4node/analyze_tempo_go_c7_lmcache_victim_abba.py" \
SLURM_JOB_ID_ENV="${SLURM_JOB_ID}" \
  "${REPO_ROOT}/.vllm_venv/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

root = Path(os.environ["RESULT_ROOT_ENV"])
runner = Path(os.environ["RUNNER_ENV"]).resolve()
analyzer = Path(os.environ["ANALYZER_ENV"]).resolve()
payload = {
    "schema": "tempo-go-c7-lmcache-victim-execution-v1",
    "slurm_job_id": int(os.environ["SLURM_JOB_ID_ENV"]),
    "allocation": "existing-approved-4node-gpu_interactive",
    "contract": os.environ["CONTRACT_ENV"],
    "contract_sha256": os.environ["CONTRACT_SHA256_ENV"],
    "runner_sha256": digest(runner),
    "analyzer_sha256": digest(analyzer),
    "batch_submission": False,
    "privileged_or_container_configuration": False,
}
(root / "execution_receipt.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

"${REPO_ROOT}/.vllm_venv/bin/python" \
  -m eval.sota_4node.analyze_tempo_go_c7_lmcache_victim_abba \
  --root "${RESULT_ROOT}" --contract "${CONTRACT}" \
  --output "${RESULT_ROOT}/analysis.json"

[[ -s "${RESULT_ROOT}/analysis.json" ]]
echo "TEMPO-GO C7 LMCache victim ABBA receipt: ${RESULT_ROOT}/analysis.json"
