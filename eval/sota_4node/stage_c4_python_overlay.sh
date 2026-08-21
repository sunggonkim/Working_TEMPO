#!/usr/bin/env bash
# Source the already prepared C4 Python bundle and fail closed if any package
# sentinel or the job-local marker differs from the prepare artifact.
set -euo pipefail

[[ $# -eq 1 ]]
: "${SLURM_JOB_ID:?Slurm job id is required}"
: "${SLURM_NODEID:?Slurm node id is required}"
: "${TEMPO_C4_PYTHON_OVERLAY_PREPARE_ARTIFACT:?prepare artifact is required}"
repo_root=$(realpath -e -- "$1")
prepare_artifact=$(realpath -e -- \
  "${TEMPO_C4_PYTHON_OVERLAY_PREPARE_ARTIFACT}")
[[ "$(jq -r '.schema' "${prepare_artifact}")" == \
  tempo-pd-c4-python-overlay-prepare-v2 ]]
[[ "$(jq -r '.slurm_job_id' "${prepare_artifact}")" == \
  "${SLURM_JOB_ID}" ]]

stage_start_ns=$(date +%s%N)
overlay=$(jq -r '.overlay' "${prepare_artifact}")
archive_sha256=$(jq -r '.archive_sha256' "${prepare_artifact}")
[[ "${overlay}" == "/tmp/tempo-c4-${SLURM_JOB_ID}-py312" ]]
[[ -f "${overlay}/.bundle-${archive_sha256}.ready" ]]
for package in transformers vllm lmcache; do
  origin="${overlay}/${package}/__init__.py"
  expected=$(jq -r --arg package "${package}" --arg origin \
    "${package}/__init__.py" \
    '.packages[] | select(.name == $package) | .sentinels[$origin]' \
    "${prepare_artifact}")
  [[ -n "${expected}" && "${expected}" != null ]]
  [[ "$(sha256sum "${origin}" | awk '{print $1}')" == "${expected}" ]]
done
stage_end_ns=$(date +%s%N)

export PYTHONPATH="${overlay}:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export TEMPO_C4_PYTHON_OVERLAY="${overlay}"
export TEMPO_C4_PYTHON_OVERLAY_ARCHIVE_SHA256="${archive_sha256}"
export TEMPO_C4_PYTHON_OVERLAY_SCHEMA=tempo-pd-c4-python-overlay-prepare-v2
export TEMPO_C4_PYTHON_OVERLAY_STAGE_ELAPSED_NS=$((stage_end_ns - stage_start_ns))
export TEMPO_C4_PYTHON_OVERLAY_PREPARE_ARTIFACT="${prepare_artifact}"

printf 'TEMPO_C4_PYTHON_OVERLAY|node=%s|path=%s|elapsed_ns=%s|archive_sha256=%s\n' \
  "${SLURM_NODEID}" "${overlay}" \
  "${TEMPO_C4_PYTHON_OVERLAY_STAGE_ELAPSED_NS}" "${archive_sha256}"
