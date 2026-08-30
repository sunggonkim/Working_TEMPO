#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "TEMPO native-allocation request refused: $*" >&2
  exit 2
}

[[ $# -eq 0 ]] || fail "arguments are not accepted"
[[ "$(id -u)" -ne 0 ]] || fail "privileged execution is forbidden"
[[ "$(command -v salloc)" == /usr/bin/salloc ]] || \
  fail "native /usr/bin/salloc required"

for tempo_go_forbidden_var in \
  SHIFTER_RUNTIME SHIFTER_IMAGE UDI CRAY_ROOTFS SLURM_CONTAINER
do
  [[ -z "${!tempo_go_forbidden_var:-}" ]] || \
    fail "forbidden container environment: ${tempo_go_forbidden_var}"
done
while IFS= read -r tempo_go_env_name; do
  case "${tempo_go_env_name}" in
    SHIFTER_*|UDI_*|SLURM_SPANK_*SHIFTER*|SLURM_SPANK_*UDI*)
      fail "forbidden container environment: ${tempo_go_env_name}"
      ;;
  esac
done < <(compgen -e)

# Fixed command by design: callers cannot append an image, container, or
# privilege-related option.  The user has granted standing approval for this
# exact 4-node, 4-hour interactive GPU allocation shape.
exec /usr/bin/salloc \
  -A m1248_g \
  -C gpu \
  -q interactive \
  -t 04:00:00 \
  -N 4 \
  --ntasks-per-node=1 \
  --cpus-per-task=128 \
  --gpus-per-node=4 \
  --network=job_vni
