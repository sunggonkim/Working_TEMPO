#!/usr/bin/env bash
# Build one reproducible Python startup bundle, cache it as a single scratch
# file, broadcast it with Slurm, and extract it to node-local /tmp.  This is a
# delivery optimization only: vLLM, LMCache, the controller, the workload, and
# the P/D transport configuration are unchanged.
set -euo pipefail

[[ $# -eq 2 ]]
: "${SLURM_JOB_ID:?Slurm job id is required}"
repo_root=$(realpath -e -- "$1")
result_dir=$(realpath -e -- "$2")
case "${result_dir}/" in "${repo_root}/results/"*) ;; *) exit 2 ;; esac

PREPARE_SRUN_JOB_ARGS=()
if [[ -n "${TEMPO_GO_C5_SRUN_JOBID:-}" ]]; then
  [[ "${TEMPO_GO_C5_SRUN_JOBID}" =~ ^[0-9]+$ ]]
  PREPARE_SRUN_JOB_ARGS=("--jobid=${TEMPO_GO_C5_SRUN_JOBID}")
fi
# This helper only creates directories, verifies a broadcast archive, and
# extracts Python files.  It never opens a communicator or moves application
# data over Slingshot.  NERSC documents no_vni for precisely this class of
# step; leaving the default VNI enabled would consume an interconnect slot
# while the co-job and native vLLM step are already using the fabric.
PREPARE_SRUN_NETWORK_ARGS=(--network=no_vni)

mapfile -t hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#hosts[@]} -eq 4 && $(printf '%s\n' "${hosts[@]}" | sort -u | wc -l) -eq 4 ]]
site_packages="${repo_root}/.vllm_venv/lib/python3.12/site-packages"
lmcache_root="${repo_root}/third_party/lmcache"
transformers_dist=transformers-5.15.0.dist-info
vllm_dist=vllm-0.26.0+cu129.dist-info
lmcache_dist=lmcache-0.1.dev1.dist-info
site_entries=(
  transformers "${transformers_dist}"
  vllm "${vllm_dist}"
  "${lmcache_dist}"
)
for entry in "${site_entries[@]}"; do
  [[ -e "${site_packages}/${entry}" ]]
done
[[ -d "${lmcache_root}/lmcache" ]]

transformers_init_sha256=$(sha256sum \
  "${site_packages}/transformers/__init__.py" | awk '{print $1}')
transformers_metadata_sha256=$(sha256sum \
  "${site_packages}/${transformers_dist}/METADATA" | awk '{print $1}')
transformers_record_sha256=$(sha256sum \
  "${site_packages}/${transformers_dist}/RECORD" | awk '{print $1}')
vllm_init_sha256=$(sha256sum \
  "${site_packages}/vllm/__init__.py" | awk '{print $1}')
vllm_metadata_sha256=$(sha256sum \
  "${site_packages}/${vllm_dist}/METADATA" | awk '{print $1}')
vllm_record_sha256=$(sha256sum \
  "${site_packages}/${vllm_dist}/RECORD" | awk '{print $1}')
lmcache_init_sha256=$(sha256sum \
  "${lmcache_root}/lmcache/__init__.py" | awk '{print $1}')
lmcache_metadata_sha256=$(sha256sum \
  "${site_packages}/${lmcache_dist}/METADATA" | awk '{print $1}')
lmcache_record_sha256=$(sha256sum \
  "${site_packages}/${lmcache_dist}/RECORD" | awk '{print $1}')
lmcache_head=$(git -C "${lmcache_root}" rev-parse HEAD)
lmcache_diff_sha256=$(git -C "${lmcache_root}" \
  diff --binary HEAD -- lmcache | sha256sum | awk '{print $1}')
mapfile -t lmcache_untracked_paths < <(git -C "${lmcache_root}" \
  ls-files --others --exclude-standard -- lmcache)
lmcache_untracked_state_sha256=$(
  for path in "${lmcache_untracked_paths[@]}"; do
    printf '%s\n' "${path}"
    sha256sum "${lmcache_root}/${path}"
  done | sha256sum | awk '{print $1}'
)
lmcache_untracked_csv=$(IFS=,; echo "${lmcache_untracked_paths[*]}")

source_key=$(printf '%s\n' \
  tempo-pd-c4-python-overlay-v2 \
  "${transformers_record_sha256}" "${vllm_record_sha256}" \
  "${lmcache_record_sha256}" "${lmcache_head}" \
  "${lmcache_diff_sha256}" "${lmcache_untracked_state_sha256}" \
  | sha256sum | awk '{print $1}')
cache_root="${repo_root}/results/tempo_pd_c4_python_overlay_cache_v2"
cache_archive="${cache_root}/${source_key}.tar"
cache_manifest="${cache_root}/${source_key}.json"
local_archive="/tmp/tempo-c4-${SLURM_JOB_ID}-${source_key}.tar"
cache_hit=0
pack_elapsed_ns=0
mkdir -p -- "${cache_root}"

if [[ -s "${cache_archive}" && -s "${cache_manifest}" ]]; then
  [[ "$(jq -r '.schema' "${cache_manifest}")" == \
    tempo-pd-c4-python-overlay-cache-v2 ]]
  [[ "$(jq -r '.source_key' "${cache_manifest}")" == "${source_key}" ]]
  archive_sha256=$(jq -r '.archive_sha256' "${cache_manifest}")
  [[ "$(sha256sum "${cache_archive}" | awk '{print $1}')" == \
    "${archive_sha256}" ]]
  cache_hit=1
else
  [[ ! -e "${cache_archive}" && ! -e "${cache_manifest}" \
    && ! -e "${local_archive}" ]]
  pack_start_ns=$(date +%s%N)
  tar -C "${site_packages}" -cf "${local_archive}" \
    "${site_entries[@]}"
  tar -C "${lmcache_root}" -rf "${local_archive}" lmcache
  pack_end_ns=$(date +%s%N)
  pack_elapsed_ns=$((pack_end_ns - pack_start_ns))
  archive_sha256=$(sha256sum "${local_archive}" | awk '{print $1}')
  cache_archive_tmp="${cache_archive}.tmp-${SLURM_JOB_ID}"
  cp -- "${local_archive}" "${cache_archive_tmp}"
  [[ "$(sha256sum "${cache_archive_tmp}" | awk '{print $1}')" == \
    "${archive_sha256}" ]]
  mv -- "${cache_archive_tmp}" "${cache_archive}"
  "${repo_root}/.vllm_venv/bin/python" - \
    "${cache_manifest}" "${source_key}" "${archive_sha256}" \
    "${cache_archive}" "${transformers_record_sha256}" \
    "${vllm_record_sha256}" "${lmcache_record_sha256}" \
    "${lmcache_head}" "${lmcache_diff_sha256}" \
    "${lmcache_untracked_state_sha256}" "${lmcache_untracked_csv}" <<'PY'
import json
import sys
from pathlib import Path

(
    output, source_key, archive_sha, archive, transformers_record,
    vllm_record, lmcache_record, lmcache_head, lmcache_diff,
    lmcache_untracked_state, lmcache_untracked_csv,
) = sys.argv[1:]
value = {
    "schema": "tempo-pd-c4-python-overlay-cache-v2",
    "source_key": source_key,
    "archive": str(Path(archive).resolve()),
    "archive_sha256": archive_sha,
    "source_state": {
        "transformers_record_sha256": transformers_record,
        "vllm_record_sha256": vllm_record,
        "lmcache_editable_record_sha256": lmcache_record,
        "lmcache_git_head": lmcache_head,
        "lmcache_tracked_diff_sha256": lmcache_diff,
        "lmcache_untracked_state_sha256": lmcache_untracked_state,
        "lmcache_untracked_files": (
            lmcache_untracked_csv.split(",") if lmcache_untracked_csv else []),
    },
}
with Path(output).open("x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY
fi

archive_bytes=$(stat -c %s "${cache_archive}")
overlay_tag=${TEMPO_C4_OVERLAY_TAG:-${SLURM_JOB_ID}}
[[ "${overlay_tag}" =~ ^[A-Za-z0-9._-]+$ ]]
overlay="/tmp/tempo-c4-${overlay_tag}-py312"
node_archive="${overlay}/.python-overlay-sbcast.tar"
marker="${overlay}/.bundle-${archive_sha256}.ready"
artifact="${result_dir}/python-overlay-prepare.json"
[[ ! -e "${artifact}" ]]

srun "${PREPARE_SRUN_JOB_ARGS[@]}" --overlap --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus=0 --cpus-per-task=1 --cpu-bind=cores \
  "${PREPARE_SRUN_NETWORK_ARGS[@]}" \
  --time=00:02:00 mkdir -p -- "${overlay}"
broadcast_start_ns=$(date +%s%N)
sbcast --force --compress --timeout=600 \
  "${cache_archive}" "${node_archive}"
broadcast_end_ns=$(date +%s%N)

export TEMPO_C4_PREPARE_OVERLAY="${overlay}"
export TEMPO_C4_PREPARE_ARCHIVE="${node_archive}"
export TEMPO_C4_PREPARE_ARCHIVE_SHA256="${archive_sha256}"
export TEMPO_C4_PREPARE_MARKER="${marker}"
export TEMPO_C4_PREPARE_TRANSFORMERS_INIT_SHA256="${transformers_init_sha256}"
export TEMPO_C4_PREPARE_TRANSFORMERS_METADATA_SHA256="${transformers_metadata_sha256}"
export TEMPO_C4_PREPARE_TRANSFORMERS_DIST="${transformers_dist}"
export TEMPO_C4_PREPARE_VLLM_INIT_SHA256="${vllm_init_sha256}"
export TEMPO_C4_PREPARE_VLLM_METADATA_SHA256="${vllm_metadata_sha256}"
export TEMPO_C4_PREPARE_VLLM_DIST="${vllm_dist}"
export TEMPO_C4_PREPARE_LMCACHE_INIT_SHA256="${lmcache_init_sha256}"
export TEMPO_C4_PREPARE_LMCACHE_METADATA_SHA256="${lmcache_metadata_sha256}"
export TEMPO_C4_PREPARE_LMCACHE_DIST="${lmcache_dist}"
srun "${PREPARE_SRUN_JOB_ARGS[@]}" --overlap --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --gpus=0 --cpus-per-task=4 --cpu-bind=cores \
  "${PREPARE_SRUN_NETWORK_ARGS[@]}" \
  --time=00:10:00 --export=ALL \
  --output="${result_dir}/python-overlay-prepare-%N.log" \
  bash -lc '
    set -euo pipefail
    extract_start_ns=$(date +%s%N)
    [[ "$(sha256sum "${TEMPO_C4_PREPARE_ARCHIVE}" | awk "{print \$1}")" == \
      "${TEMPO_C4_PREPARE_ARCHIVE_SHA256}" ]]
    tar -C "${TEMPO_C4_PREPARE_OVERLAY}" -xf \
      "${TEMPO_C4_PREPARE_ARCHIVE}"
    [[ "$(sha256sum "${TEMPO_C4_PREPARE_OVERLAY}/transformers/__init__.py" | awk "{print \$1}")" == \
      "${TEMPO_C4_PREPARE_TRANSFORMERS_INIT_SHA256}" ]]
    [[ "$(sha256sum "${TEMPO_C4_PREPARE_OVERLAY}/${TEMPO_C4_PREPARE_TRANSFORMERS_DIST}/METADATA" | awk "{print \$1}")" == \
      "${TEMPO_C4_PREPARE_TRANSFORMERS_METADATA_SHA256}" ]]
    [[ "$(sha256sum "${TEMPO_C4_PREPARE_OVERLAY}/vllm/__init__.py" | awk "{print \$1}")" == \
      "${TEMPO_C4_PREPARE_VLLM_INIT_SHA256}" ]]
    [[ "$(sha256sum "${TEMPO_C4_PREPARE_OVERLAY}/${TEMPO_C4_PREPARE_VLLM_DIST}/METADATA" | awk "{print \$1}")" == \
      "${TEMPO_C4_PREPARE_VLLM_METADATA_SHA256}" ]]
    [[ "$(sha256sum "${TEMPO_C4_PREPARE_OVERLAY}/lmcache/__init__.py" | awk "{print \$1}")" == \
      "${TEMPO_C4_PREPARE_LMCACHE_INIT_SHA256}" ]]
    [[ "$(sha256sum "${TEMPO_C4_PREPARE_OVERLAY}/${TEMPO_C4_PREPARE_LMCACHE_DIST}/METADATA" | awk "{print \$1}")" == \
      "${TEMPO_C4_PREPARE_LMCACHE_METADATA_SHA256}" ]]
    touch -- "${TEMPO_C4_PREPARE_MARKER}"
    extract_end_ns=$(date +%s%N)
    printf "TEMPO_C4_SBCAST_OVERLAY|node_id=%s|host=%s|elapsed_ns=%s|archive_sha256=%s\\n" \
      "${SLURM_NODEID}" "$(hostname)" \
      "$((extract_end_ns - extract_start_ns))" \
      "${TEMPO_C4_PREPARE_ARCHIVE_SHA256}"
  '

hosts_csv=$(IFS=,; echo "${hosts[*]}")
"${repo_root}/.vllm_venv/bin/python" - \
  "${artifact}" "${SLURM_JOB_ID}" "${overlay}" "${hosts_csv}" \
  "${archive_sha256}" "${archive_bytes}" "${pack_elapsed_ns}" \
  "$((broadcast_end_ns - broadcast_start_ns))" "${result_dir}" \
  "${cache_hit}" "${cache_archive}" "${cache_manifest}" \
  "${source_key}" "${transformers_init_sha256}" \
  "${transformers_metadata_sha256}" "${transformers_dist}" \
  "${vllm_init_sha256}" "${vllm_metadata_sha256}" "${vllm_dist}" \
  "${lmcache_init_sha256}" "${lmcache_metadata_sha256}" \
  "${lmcache_dist}" "${lmcache_head}" "${lmcache_diff_sha256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    artifact_raw, job_id, overlay, hosts_csv, archive_sha, archive_bytes,
    pack_ns, broadcast_ns, result_dir_raw, cache_hit, cache_archive,
    cache_manifest, source_key, transformers_init, transformers_metadata,
    transformers_dist, vllm_init, vllm_metadata, vllm_dist, lmcache_init,
    lmcache_metadata, lmcache_dist, lmcache_head, lmcache_diff,
) = sys.argv[1:]
result_dir = Path(result_dir_raw)
logs = []
for host in hosts_csv.split(","):
    path = result_dir / f"python-overlay-prepare-{host}.log"
    payload = path.read_bytes()
    logs.append({
        "host": host,
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "line": payload.decode("utf-8").strip(),
    })
packages = [
    {
        "name": "transformers",
        "dist_info": transformers_dist,
        "sentinels": {
            "transformers/__init__.py": transformers_init,
            f"{transformers_dist}/METADATA": transformers_metadata,
        },
    },
    {
        "name": "vllm",
        "dist_info": vllm_dist,
        "sentinels": {
            "vllm/__init__.py": vllm_init,
            f"{vllm_dist}/METADATA": vllm_metadata,
        },
    },
    {
        "name": "lmcache",
        "dist_info": lmcache_dist,
        "sentinels": {
            "lmcache/__init__.py": lmcache_init,
            f"{lmcache_dist}/METADATA": lmcache_metadata,
        },
        "git_head": lmcache_head,
        "tracked_diff_sha256": lmcache_diff,
    },
]
value = {
    "schema": "tempo-pd-c4-python-overlay-prepare-v2",
    "slurm_job_id": job_id,
    "hosts": hosts_csv.split(","),
    "overlay": overlay,
    "packages": packages,
    "source_key": source_key,
    "archive_sha256": archive_sha,
    "archive_bytes": int(archive_bytes),
    "cache_hit": cache_hit == "1",
    "cache_archive": str(Path(cache_archive).resolve()),
    "cache_manifest": str(Path(cache_manifest).resolve()),
    "pack_elapsed_ns": int(pack_ns),
    "broadcast_elapsed_ns": int(broadcast_ns),
    "node_extract_logs": logs,
    "delivery": "one_versioned_archive_then_slurm_sbcast_to_node_local_tmp",
    "delivery_only": True,
    "controller_or_workload_changed": False,
    "pd_data_plane_changed": False,
}
path = Path(artifact_raw)
with path.open("x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

printf 'TEMPO_C4_SBCAST_PREPARE|artifact=%s|cache_hit=%s|pack_ns=%s|broadcast_ns=%s|bytes=%s\n' \
  "${artifact}" "${cache_hit}" "${pack_elapsed_ns}" \
  "$((broadcast_end_ns - broadcast_start_ns))" "${archive_bytes}"
