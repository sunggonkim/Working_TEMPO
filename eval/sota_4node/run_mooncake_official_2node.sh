#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_WHEEL_VERSION="0.3.12.post1"
readonly BLOCK_SIZE_BYTES="33554432"
readonly BUFFER_SIZE_BYTES="134217728"
readonly THREADS="4"
readonly BATCH_SIZE="1"
readonly DURATION_SECONDS="5"
readonly TARGET_LIFETIME_SECONDS="18"
readonly STEP_LIFETIME_SECONDS="25"

if [[ $# -ne 1 ]]; then
    echo "usage: $0 RESULT_DIR" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PARSER="${SCRIPT_DIR}/parse_mooncake_official.py"
VENV_PYTHON="${REPO_ROOT}/.sota_venv/bin/python"
BENCH="${REPO_ROOT}/.sota_venv/bin/transfer_engine_bench"
MOONCAKE_REPO="${REPO_ROOT}/third_party/mooncake"

if [[ $1 == /* ]]; then
    RESULT_CANDIDATE=$1
else
    RESULT_CANDIDATE="${REPO_ROOT}/$1"
fi
RESULT_DIR=$(realpath -m -- "${RESULT_CANDIDATE}")
case "${RESULT_DIR}/" in
    "${REPO_ROOT}/"*) ;;
    *)
        echo "RESULT_DIR must be inside ${REPO_ROOT}" >&2
        exit 2
        ;;
esac
if [[ "${RESULT_DIR}" == "${REPO_ROOT}" ]]; then
    echo "RESULT_DIR must not be the repository root" >&2
    exit 2
fi

: "${SLURM_JOB_ID:?run inside an existing two-node allocation}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"
if [[ ${SLURM_JOB_NUM_NODES:-0} -ne 2 ]]; then
    echo "this launcher requires an existing two-node allocation" >&2
    exit 2
fi

module reset
module load pytorch/2.8.0
export PYTHONSAFEPATH=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

[[ -x "${VENV_PYTHON}" ]]
[[ -x "${BENCH}" ]]
[[ -f "${PARSER}" ]]
[[ -d "${MOONCAKE_REPO}/.git" ]]

CUDA_RUNTIME_LIB=$("${VENV_PYTHON}" -c \
    "import importlib.metadata as m; print(m.distribution('nvidia-cuda-runtime-cu12').locate_file('nvidia/cuda_runtime/lib'))")
if [[ ! -d "${CUDA_RUNTIME_LIB}" || ! -r "${CUDA_RUNTIME_LIB}/libcudart.so.12" ]]; then
    echo "CUDA runtime wheel does not provide readable libcudart.so.12" >&2
    exit 2
fi
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
set +e
BENCH_HELP_OUTPUT=$("${BENCH}" --help 2>&1)
BENCH_HELP_RC=$?
set -e
if [[ ${BENCH_HELP_RC} -ne 0 && ${BENCH_HELP_RC} -ne 1 ]] ||
    [[ "${BENCH_HELP_OUTPUT}" != *"Transfer protocol:"* ]]; then
    echo "Mooncake transfer_engine_bench preflight failed" >&2
    exit 2
fi

WHEEL_VERSION=$("${VENV_PYTHON}" -c \
    "import importlib.metadata as m; print(m.version('mooncake-transfer-engine'))")
if [[ "${WHEEL_VERSION}" != "${EXPECTED_WHEEL_VERSION}" ]]; then
    echo "expected Mooncake ${EXPECTED_WHEEL_VERSION}, got ${WHEEL_VERSION}" >&2
    exit 2
fi
MOONCAKE_COMMIT=$(git -C "${MOONCAKE_REPO}" rev-parse HEAD)
MOONCAKE_ORIGIN=$(git -C "${MOONCAKE_REPO}" remote get-url origin)

mapfile -t ALLOCATION_NODES < <(
    scontrol show hostnames "${SLURM_JOB_NODELIST}"
)
if [[ ${#ALLOCATION_NODES[@]} -ne 2 ]]; then
    echo "allocation must resolve to exactly two nodes" >&2
    exit 2
fi
TARGET_HOST=${ALLOCATION_NODES[0]}
RPC_PORT=$((15000 + SLURM_JOB_ID % 3000))
INITIATOR_RPC_PORT=$((RPC_PORT + 1))

mkdir -p -- "${RESULT_DIR}"
MANIFEST="${RESULT_DIR}/manifest.json"
RESULT_JSON="${RESULT_DIR}/result.json"

"${VENV_PYTHON}" "${PARSER}" manifest \
    --output "${MANIFEST}" \
    --wheel-version "${WHEEL_VERSION}" \
    --git-commit "${MOONCAKE_COMMIT}" \
    --git-repository "${MOONCAKE_ORIGIN}" \
    --binary "${BENCH}" \
    --job-id "${SLURM_JOB_ID}" \
    --node-list "${SLURM_JOB_NODELIST}"

timeout --signal=TERM --kill-after=3s "${STEP_LIFETIME_SECONDS}s" \
    srun --exact \
    --nodes=2 \
    --ntasks=2 \
    --ntasks-per-node=1 \
    --cpus-per-task=32 \
    --gpus-per-task=4 \
    --gpu-bind=none \
    --kill-on-bad-exit=1 \
    --output="${RESULT_DIR}/rank-%t.stdout.log" \
    --error="${RESULT_DIR}/rank-%t.stderr.log" \
    bash --noprofile --norc -c '
        set -euo pipefail
        bench=$1
        target_host=$2
        rpc_port=$3
        initiator_rpc_port=$4
        target_lifetime=$5
        block_size=$6
        buffer_size=$7
        threads=$8
        batch_size=$9
        duration=${10}
        local_host=$(hostname -s)

        common_flags=(
            --metadata_server=P2PHANDSHAKE
            --backend=classic
            --protocol=tcp
            --auto_discovery=false
            --use_vram=true
            --gpu_id=-1
            --operation=read
            --block_size="${block_size}"
            --buffer_size="${buffer_size}"
            --threads="${threads}"
            --batch_size="${batch_size}"
            --duration="${duration}"
            --report_unit=GB
            --report_precision=6
        )

        if [[ ${SLURM_PROCID} -eq 0 ]]; then
            set +e
            timeout --signal=TERM --kill-after=2s "${target_lifetime}s" \
                "${bench}" \
                --mode=target \
                --local_server_name="${target_host}:${rpc_port}" \
                "${common_flags[@]}"
            target_rc=$?
            set -e
            if [[ ${target_rc} -eq 124 || ${target_rc} -eq 143 ]]; then
                exit 0
            fi
            exit "${target_rc}"
        fi

        if [[ ${SLURM_PROCID} -ne 1 ]]; then
            echo "unexpected task rank ${SLURM_PROCID}" >&2
            exit 2
        fi
        sleep 5
        exec "${bench}" \
            --mode=initiator \
            --local_server_name="${local_host}:${initiator_rpc_port}" \
            --segment_id="${target_host}:${rpc_port}" \
            "${common_flags[@]}"
    ' bash \
    "${BENCH}" \
    "${TARGET_HOST}" \
    "${RPC_PORT}" \
    "${INITIATOR_RPC_PORT}" \
    "${TARGET_LIFETIME_SECONDS}" \
    "${BLOCK_SIZE_BYTES}" \
    "${BUFFER_SIZE_BYTES}" \
    "${THREADS}" \
    "${BATCH_SIZE}" \
    "${DURATION_SECONDS}"

"${VENV_PYTHON}" "${PARSER}" result \
    --manifest "${MANIFEST}" \
    --initiator-log "${RESULT_DIR}/rank-1.stderr.log" \
    --initiator-log "${RESULT_DIR}/rank-1.stdout.log" \
    --target-log "${RESULT_DIR}/rank-0.stderr.log" \
    --target-log "${RESULT_DIR}/rank-0.stdout.log" \
    --output "${RESULT_JSON}"

echo "Mooncake official Transfer Engine result: ${RESULT_JSON}"
