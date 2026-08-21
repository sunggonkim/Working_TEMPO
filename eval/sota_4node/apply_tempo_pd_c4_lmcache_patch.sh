#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
lmcache_root="${repo_root}/third_party/lmcache"
patch_path="${script_dir}/lmcache_tempo_c4_runtime.patch"
expected_head="227d13f5c9fdb52ddb933641d34331f678de03a0"

git -C "${lmcache_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
  echo "LMCache checkout missing at ${lmcache_root}" >&2
  exit 2
}
actual_head="$(git -C "${lmcache_root}" rev-parse HEAD)"
test "${actual_head}" = "${expected_head}" || {
  echo "LMCache HEAD mismatch: ${actual_head}" >&2
  exit 2
}
git -C "${lmcache_root}" apply --check "${patch_path}"
git -C "${lmcache_root}" apply "${patch_path}"
(
  cd "${repo_root}"
  sha256sum --check - <<'SHA256'
7750b8d2d13474db0f6b5c5eb8920c21556fa936d33d3b4f01cacfa448289988  third_party/lmcache/examples/disagg_prefill/disagg_proxy_server.py
3ade8f5731735331f87ea20f2cf1d111f34e9d45f94fd77fc47b02dd4dc742fb  third_party/lmcache/lmcache/integration/vllm/vllm_v1_adapter.py
550f2799ad2322ef6a493ccb0625e05f2e64c36e72de0ed544891b68ff53aa93  third_party/lmcache/lmcache/v1/cache_engine.py
54cd9db454bf5f05624985a09a20015856cb93516af1d0133f8f83ed0ea445bb  third_party/lmcache/lmcache/v1/storage_backend/pd_backend_async.py
38191b278510cc4357dc04aba54a2a31c9e4eebe94156a016c0ceedd12bd1f7f  third_party/lmcache/lmcache/v1/transfer_channel/tempo_nixl_hotpath.py
46c8fb90d71aa494b675f93b1185a0659119eb234f75bc5954fdccdeaf1ce0d4  third_party/lmcache/lmcache/v1/transfer_channel/tempo_nixl_hotpath_v2.py
SHA256
)
echo "LMCache C4 runtime patch applied and verified"
