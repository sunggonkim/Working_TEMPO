#!/usr/bin/env bash
# Source this file before launching the C4 Python process.  Transformers 5.x
# discovers hundreds of model modules at import time; keeping that metadata on
# node-local storage avoids a four-node DVS metadata storm without changing the
# vLLM/LMCache data plane or any frozen C4 input.
set -euo pipefail

[[ $# -eq 1 ]]
: "${SLURM_JOB_ID:?Slurm job id is required}"
: "${SLURM_NODEID:?Slurm node id is required}"

repo_root=$(realpath -e -- "$1")
site_packages="${repo_root}/.vllm_venv/lib/python3.12/site-packages"
package_name=transformers
dist_name=transformers-5.15.0.dist-info
source_init="${site_packages}/${package_name}/__init__.py"
source_metadata="${site_packages}/${dist_name}/METADATA"
[[ -f "${source_init}" && -f "${source_metadata}" ]]

source_init_sha256=$(sha256sum "${source_init}" | awk '{print $1}')
source_metadata_sha256=$(sha256sum "${source_metadata}" | awk '{print $1}')
overlay_tag=${TEMPO_C4_OVERLAY_TAG:-${SLURM_JOB_ID}}
[[ "${overlay_tag}" =~ ^[A-Za-z0-9._-]+$ ]]
overlay="/tmp/tempo-c4-${overlay_tag}-py312"
marker="${overlay}/.transformers-${source_init_sha256}-${source_metadata_sha256}.ready"
archive="${overlay}/.transformers-${SLURM_NODEID}.tar"
mkdir -p -- "${overlay}"

stage_start_ns=$(date +%s%N)
staged_this_entry=0
archive_bytes=0
if [[ "${TEMPO_C4_PYTHON_OVERLAY_REQUIRE_PREPARED:-0}" == 1 ]]; then
  [[ -f "${marker}" ]]
elif [[ ! -f "${marker}" ]]; then
  staged_this_entry=1
  tar -C "${site_packages}" -cf "${archive}" \
    "${package_name}" "${dist_name}"
  archive_bytes=$(stat -c %s "${archive}")
  tar -C "${overlay}" -xf "${archive}"
  [[ "$(sha256sum "${overlay}/${package_name}/__init__.py" | awk '{print $1}')" == \
    "${source_init_sha256}" ]]
  [[ "$(sha256sum "${overlay}/${dist_name}/METADATA" | awk '{print $1}')" == \
    "${source_metadata_sha256}" ]]
  touch -- "${marker}"
  rm -f -- "${archive}"
fi
stage_end_ns=$(date +%s%N)

export PYTHONPATH="${overlay}${PYTHONPATH:+:${PYTHONPATH}}"
export TEMPO_C4_PYTHON_OVERLAY="${overlay}"
export TEMPO_C4_PYTHON_OVERLAY_PACKAGE="${package_name}"
export TEMPO_C4_PYTHON_OVERLAY_DIST_INFO="${dist_name}"
export TEMPO_C4_PYTHON_OVERLAY_SOURCE_INIT_SHA256="${source_init_sha256}"
export TEMPO_C4_PYTHON_OVERLAY_SOURCE_METADATA_SHA256="${source_metadata_sha256}"
export TEMPO_C4_PYTHON_OVERLAY_STAGE_ELAPSED_NS=$((stage_end_ns - stage_start_ns))
export TEMPO_C4_PYTHON_OVERLAY_ARCHIVE_BYTES="${archive_bytes}"
export TEMPO_C4_PYTHON_OVERLAY_STAGED_THIS_ENTRY="${staged_this_entry}"
export TEMPO_C4_PYTHON_OVERLAY_PREPARE_ARTIFACT="${TEMPO_C4_PYTHON_OVERLAY_PREPARE_ARTIFACT:-}"

printf 'TEMPO_C4_PYTHON_OVERLAY|node=%s|path=%s|staged=%s|elapsed_ns=%s|archive_bytes=%s\n' \
  "${SLURM_NODEID}" "${overlay}" "${staged_this_entry}" \
  "${TEMPO_C4_PYTHON_OVERLAY_STAGE_ELAPSED_NS}" "${archive_bytes}"
