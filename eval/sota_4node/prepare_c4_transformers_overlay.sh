#!/usr/bin/env bash
# Package Transformers once on the allocation leader, broadcast the archive
# with Slurm, and extract it on every allocated node.  This avoids four
# simultaneous recursive metadata walks over Perlmutter DVS.
set -euo pipefail

[[ $# -eq 2 ]]
: "${SLURM_JOB_ID:?Slurm job id is required}"
repo_root=$(realpath -e -- "$1")
result_dir=$(realpath -e -- "$2")
case "${result_dir}/" in "${repo_root}/results/"*) ;; *) exit 2 ;; esac

mapfile -t hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
[[ ${#hosts[@]} -eq 4 && $(printf '%s\n' "${hosts[@]}" | sort -u | wc -l) -eq 4 ]]
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
source_archive="/tmp/tempo-c4-${overlay_tag}-transformers-source.tar"
node_archive="${overlay}/.transformers-sbcast.tar"
marker="${overlay}/.transformers-${source_init_sha256}-${source_metadata_sha256}.ready"
artifact="${result_dir}/python-overlay-prepare.json"
[[ ! -e "${artifact}" && ! -e "${source_archive}" ]]

pack_start_ns=$(date +%s%N)
tar -C "${site_packages}" -cf "${source_archive}" \
  "${package_name}" "${dist_name}"
pack_end_ns=$(date +%s%N)
archive_bytes=$(stat -c %s "${source_archive}")
archive_sha256=$(sha256sum "${source_archive}" | awk '{print $1}')

srun --overlap --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --cpus-per-task=1 --cpu-bind=cores \
  --time=00:02:00 mkdir -p -- "${overlay}"
broadcast_start_ns=$(date +%s%N)
sbcast --force --compress --timeout=300 \
  "${source_archive}" "${node_archive}"
broadcast_end_ns=$(date +%s%N)

export TEMPO_C4_PREPARE_OVERLAY="${overlay}"
export TEMPO_C4_PREPARE_ARCHIVE="${node_archive}"
export TEMPO_C4_PREPARE_ARCHIVE_SHA256="${archive_sha256}"
export TEMPO_C4_PREPARE_PACKAGE="${package_name}"
export TEMPO_C4_PREPARE_DIST_INFO="${dist_name}"
export TEMPO_C4_PREPARE_INIT_SHA256="${source_init_sha256}"
export TEMPO_C4_PREPARE_METADATA_SHA256="${source_metadata_sha256}"
export TEMPO_C4_PREPARE_MARKER="${marker}"
srun --overlap --exact --nodes=4 --ntasks=4 --ntasks-per-node=1 \
  --distribution=block:block --cpus-per-task=4 --cpu-bind=cores \
  --time=00:10:00 --export=ALL \
  --output="${result_dir}/python-overlay-prepare-%N.log" \
  bash -lc '
    set -euo pipefail
    extract_start_ns=$(date +%s%N)
    [[ "$(sha256sum "${TEMPO_C4_PREPARE_ARCHIVE}" | awk "{print \$1}")" == \
      "${TEMPO_C4_PREPARE_ARCHIVE_SHA256}" ]]
    tar -C "${TEMPO_C4_PREPARE_OVERLAY}" -xf \
      "${TEMPO_C4_PREPARE_ARCHIVE}"
    [[ "$(sha256sum "${TEMPO_C4_PREPARE_OVERLAY}/${TEMPO_C4_PREPARE_PACKAGE}/__init__.py" | awk "{print \$1}")" == \
      "${TEMPO_C4_PREPARE_INIT_SHA256}" ]]
    [[ "$(sha256sum "${TEMPO_C4_PREPARE_OVERLAY}/${TEMPO_C4_PREPARE_DIST_INFO}/METADATA" | awk "{print \$1}")" == \
      "${TEMPO_C4_PREPARE_METADATA_SHA256}" ]]
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
  "${package_name}" "${dist_name}" "${source_init_sha256}" \
  "${source_metadata_sha256}" "${archive_sha256}" "${archive_bytes}" \
  "$((pack_end_ns - pack_start_ns))" \
  "$((broadcast_end_ns - broadcast_start_ns))" "${result_dir}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    artifact_raw, job_id, overlay, hosts_csv, package, dist_info,
    init_sha, metadata_sha, archive_sha, archive_bytes, pack_ns,
    broadcast_ns, result_dir_raw,
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
value = {
    "schema": "tempo-pd-c4-python-overlay-prepare-v1",
    "slurm_job_id": job_id,
    "hosts": hosts_csv.split(","),
    "overlay": overlay,
    "package": package,
    "dist_info": dist_info,
    "source_init_sha256": init_sha,
    "source_metadata_sha256": metadata_sha,
    "archive_sha256": archive_sha,
    "archive_bytes": int(archive_bytes),
    "pack_elapsed_ns": int(pack_ns),
    "broadcast_elapsed_ns": int(broadcast_ns),
    "node_extract_logs": logs,
    "delivery": "single_dvs_pack_then_slurm_sbcast_to_node_local_tmp",
    "delivery_only": True,
    "controller_or_workload_changed": False,
    "pd_data_plane_changed": False,
}
path = Path(artifact_raw)
with path.open("x", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
PY

printf 'TEMPO_C4_SBCAST_PREPARE|artifact=%s|pack_ns=%s|broadcast_ns=%s|bytes=%s\n' \
  "${artifact}" "$((pack_end_ns - pack_start_ns))" \
  "$((broadcast_end_ns - broadcast_start_ns))" "${archive_bytes}"
