#!/bin/bash
# Reproduce the pinned open-source baseline plus the v2 -> v3 -> v4 patch chain
# used by the one-shot Perlmutter gate.
# Run from the repository root after: module load pytorch/2.8.0

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_ROOT="${REPO_ROOT}/.baseline_build"
DEPS_ROOT="${REPO_ROOT}/.baseline_deps"
RUNTIME="${DEPS_ROOT}/runtime"
mkdir -p "${BUILD_ROOT}" "${DEPS_ROOT}" "${RUNTIME}"

# Fail before any clone/build work when the Perlmutter software stack is not
# the one used to build the pinned extension.
python -c 'import sys, torch, msgpack; assert sys.version_info[:2] == (3, 12), sys.version; assert torch.__version__ == "2.8.0+cu129", (torch.__version__, torch.__file__); assert torch.version.cuda == "12.9", torch.version.cuda; assert msgpack.__version__ == "1.1.1", (msgpack.__version__, msgpack.__file__)'

TORCHSNAPSHOT_COMMIT=a0ff1ec289c08f6cf6408c544b6ad23605c93462
DATASTATES_COMMIT=1b849900f402cff614ff3d7e62e3bb29cfb29fa4
LIBURING_COMMIT=08468cc3830185c75f9e7edefd88aa01e5c2f8ab

clone_at_commit() {
    local repository=$1
    local destination=$2
    local commit=$3
    local allow_dirty=${4:-0}
    if [[ ! -d "${destination}/.git" ]]; then
        if [[ -e "${destination}" && ! -d "${destination}" ]]; then
            echo "ERROR: clone destination is not a directory: ${destination}" >&2
            return 2
        fi
        git clone "${repository}" "${destination}"
    fi
    local current
    current=$(git -C "${destination}" rev-parse HEAD)
    if [[ "${current}" != "${commit}" \
          && -n "$(git -C "${destination}" status --porcelain)" ]]; then
        echo "ERROR: refusing to move a dirty checkout at ${destination}" >&2
        return 2
    fi
    git -C "${destination}" fetch origin "${commit}"
    if [[ "${current}" != "${commit}" ]]; then
        git -C "${destination}" checkout --detach "${commit}"
    fi
    test "$(git -C "${destination}" rev-parse HEAD)" = "${commit}"
    if [[ "${allow_dirty}" != "1" \
          && -n "$(git -C "${destination}" status --porcelain)" ]]; then
        echo "ERROR: pinned dependency checkout is dirty: ${destination}" >&2
        return 2
    fi
}

clone_at_commit https://github.com/pytorch/torchsnapshot.git "${BUILD_ROOT}/torchsnapshot" "${TORCHSNAPSHOT_COMMIT}"
clone_at_commit https://github.com/DataStates/datastates-llm.git "${BUILD_ROOT}/datastates-llm" "${DATASTATES_COMMIT}" 1
clone_at_commit https://github.com/axboe/liburing.git "${BUILD_ROOT}/liburing" "${LIBURING_COMMIT}"

TEMPO_V2_PATCH="${REPO_ROOT}/eval/sota_4node/datastates_tempo_v2.patch"
TEMPO_V3_PATCH="${REPO_ROOT}/eval/sota_4node/datastates_tempo_v3.patch"
TEMPO_V4_PATCH="${REPO_ROOT}/eval/sota_4node/datastates_tempo_v4.patch"
test -s "${TEMPO_V2_PATCH}"
test -s "${TEMPO_V3_PATCH}"
test -s "${TEMPO_V4_PATCH}"

# Reconstruct every valid prefix in a temporary repository, then compare Git
# tree identities through a temporary index.  This ignores generated files but
# catches extra source edits even inside patch-touched files.  An unexpected
# dirty checkout is rejected before applying anything; the shared checkout is
# never reset or cleaned.
VERIFY_ROOT=""
cleanup_verify_root() {
    if [[ -n "${VERIFY_ROOT}" && -d "${VERIFY_ROOT}" \
          && "${VERIFY_ROOT}" =~ ^${BUILD_ROOT}/tempo-patch-verify\.[^/]+$ ]]; then
        find "${VERIFY_ROOT}" -depth -delete
        rmdir -- "${VERIFY_ROOT}" 2>/dev/null || true
    fi
}
trap cleanup_verify_root EXIT
VERIFY_ROOT=$(mktemp -d "${BUILD_ROOT}/tempo-patch-verify.XXXXXX")
git -C "${BUILD_ROOT}/datastates-llm" archive "${DATASTATES_COMMIT}" | tar -xf - -C "${VERIFY_ROOT}"
git -C "${VERIFY_ROOT}" init -q
git -C "${VERIFY_ROOT}" add -A
base_tree=$(git -C "${VERIFY_ROOT}" write-tree)
git -C "${VERIFY_ROOT}" apply --check "${TEMPO_V2_PATCH}"
git -C "${VERIFY_ROOT}" apply "${TEMPO_V2_PATCH}"
git -C "${VERIFY_ROOT}" add -A
v2_tree=$(git -C "${VERIFY_ROOT}" write-tree)
git -C "${VERIFY_ROOT}" apply --check "${TEMPO_V3_PATCH}"
git -C "${VERIFY_ROOT}" apply "${TEMPO_V3_PATCH}"
git -C "${VERIFY_ROOT}" add -A
v3_tree=$(git -C "${VERIFY_ROOT}" write-tree)
git -C "${VERIFY_ROOT}" apply --check "${TEMPO_V4_PATCH}"
git -C "${VERIFY_ROOT}" apply "${TEMPO_V4_PATCH}"
git -C "${VERIFY_ROOT}" add -A
v4_tree=$(git -C "${VERIFY_ROOT}" write-tree)
actual_index="${VERIFY_ROOT}/.git/actual-index"
GIT_INDEX_FILE="${actual_index}" git -C "${BUILD_ROOT}/datastates-llm" read-tree "${DATASTATES_COMMIT}"
GIT_INDEX_FILE="${actual_index}" git -C "${BUILD_ROOT}/datastates-llm" add -A
actual_tree=$(GIT_INDEX_FILE="${actual_index}" git -C "${BUILD_ROOT}/datastates-llm" write-tree)

apply_checked() {
    local patch=$1
    git -C "${BUILD_ROOT}/datastates-llm" apply --check "${patch}"
    git -C "${BUILD_ROOT}/datastates-llm" apply "${patch}"
}
case "${actual_tree}" in
    "${base_tree}")
        apply_checked "${TEMPO_V2_PATCH}"
        apply_checked "${TEMPO_V3_PATCH}"
        apply_checked "${TEMPO_V4_PATCH}"
        ;;
    "${v2_tree}")
        apply_checked "${TEMPO_V3_PATCH}"
        apply_checked "${TEMPO_V4_PATCH}"
        ;;
    "${v3_tree}")
        apply_checked "${TEMPO_V4_PATCH}"
        ;;
    "${v4_tree}")
        echo "TEMPO v2+v3+v4 DataStates patches already applied"
        ;;
    *)
        echo "ERROR: refusing a DataStates checkout outside an exact patch-chain prefix" >&2
        exit 2
        ;;
esac
GIT_INDEX_FILE="${actual_index}" git -C "${BUILD_ROOT}/datastates-llm" read-tree "${DATASTATES_COMMIT}"
GIT_INDEX_FILE="${actual_index}" git -C "${BUILD_ROOT}/datastates-llm" add -A
actual_tree=$(GIT_INDEX_FILE="${actual_index}" git -C "${BUILD_ROOT}/datastates-llm" write-tree)
test "${actual_tree}" = "${v4_tree}"
git -C "${BUILD_ROOT}/datastates-llm" diff --check
cleanup_verify_root
VERIFY_ROOT=""

python -m pip install --no-deps --upgrade --target "${RUNTIME}" \
    "${BUILD_ROOT}/torchsnapshot" "${BUILD_ROOT}/datastates-llm/llm"
PYTHON_DEPENDENCIES=(
    PyYAML==6.0.3
    aiofiles==25.1.0
    aiohttp==3.14.3
    importlib-metadata==9.0.0
    psutil==7.2.2
    pyre-extensions==0.0.32
    typing-extensions==4.16.0
    nanobind==2.13.0
    fasteners==0.20
    msgpack==1.1.1
)
python -m pip install --upgrade --target "${RUNTIME}" "${PYTHON_DEPENDENCIES[@]}"

(
    cd "${BUILD_ROOT}/liburing"
    ./configure --prefix="${DEPS_ROOT}/liburing"
    make -j8 library
    make install
)

export CC=/opt/cray/pe/gcc-native/13/bin/gcc
export CXX=/opt/cray/pe/gcc-native/13/bin/g++
export PYTHONPATH="${RUNTIME}:${PYTHONPATH:-}"
cmake -S "${BUILD_ROOT}/datastates-llm" -B "${BUILD_ROOT}/datastates-build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_INSTALL_PREFIX="${DEPS_ROOT}/datastates" \
    -DLIBURING_ROOT="${DEPS_ROOT}/liburing" \
    -Dnanobind_DIR="${RUNTIME}/nanobind/cmake"
cmake --build "${BUILD_ROOT}/datastates-build" \
    --target datastates_core_py test_tempo_credit_controller \
        test_logical_file_layout -j8
ctest --test-dir "${BUILD_ROOT}/datastates-build/tests" \
    --output-on-failure \
    -R '^cpp_test_(tempo_credit_controller|logical_file_layout)$'
cmake --install "${BUILD_ROOT}/datastates-build" --prefix "${DEPS_ROOT}/datastates"
cp -a "${DEPS_ROOT}/datastates/datastates/datastates_core.cpython-312-x86_64-linux-gnu.so" "${RUNTIME}/datastates/"
mkdir -p "${RUNTIME}/datastates/lib"
cp -a "${DEPS_ROOT}/datastates/datastates/lib/." "${RUNTIME}/datastates/lib/"

# Region identities must come from one process-wide allocator in the core SO.
# A header-local counter is duplicated by the nanobind DSO under hidden
# visibility and corrupts io_uring accounting when payload and metadata UIDs
# collide.  Verify both sides of the dynamic-link contract explicitly.
nm -D --defined-only "${RUNTIME}/datastates/lib/libdatastates_core.so" \
    | c++filt | rg 'datastates::next_internal_uid\(\)' >/dev/null
nm -D --undefined-only \
    "${RUNTIME}/datastates/datastates_core.cpython-312-x86_64-linux-gnu.so" \
    | c++filt | rg 'datastates::next_internal_uid\(\)' >/dev/null

export LD_LIBRARY_PATH="${RUNTIME}/datastates/lib:${DEPS_ROOT}/liburing/lib:${LD_LIBRARY_PATH:-}"
python - <<'PY'
import torch
import torchsnapshot
import datastates
import msgpack
from importlib.metadata import version
from datastates import datastates_core

assert torch.__version__ == "2.8.0+cu129", (torch.__version__, torch.__file__)
expected_dependencies = {
    "PyYAML": "6.0.3",
    "aiofiles": "25.1.0",
    "aiohttp": "3.14.3",
    "importlib-metadata": "9.0.0",
    "psutil": "7.2.2",
    "pyre-extensions": "0.0.32",
    "typing-extensions": "4.16.0",
    "nanobind": "2.13.0",
    "fasteners": "0.20",
    "msgpack": "1.1.1",
}
assert {name: version(name) for name in expected_dependencies} == expected_dependencies
assert msgpack.__version__ == "1.1.1", (msgpack.__version__, msgpack.__file__)
required = (
    "ckpt", "restore", "wait", "configure_d2h_pacing", "set_d2h_paused",
    "set_persistence_paused", "get_d2h_first_issue_unix_ns",
    "install_credit_plan", "prepare_credit_transition", "enqueue_credit_transition",
    "pending_credit_transition_callbacks", "retire_credit_transition_callbacks",
    "get_last_checkpoint_layout",
    "get_stage_stats", "take_admission_trace", "force_drain",
    "disable_credit_control", "get_queue_stats", "shutdown",
)
missing = [name for name in required if not hasattr(datastates_core.state_io_engine, name)]
assert not missing, missing
PY
MANIFEST="${RUNTIME}/datastates_build_manifest.txt"
DATASTATES_SO="${RUNTIME}/datastates/datastates_core.cpython-312-x86_64-linux-gnu.so"
DATASTATES_CORE_SO="${RUNTIME}/datastates/lib/libdatastates_core.so"
{
    printf 'torchsnapshot_commit=%s\n' "${TORCHSNAPSHOT_COMMIT}"
    printf 'datastates_commit=%s\n' "${DATASTATES_COMMIT}"
    printf 'liburing_commit=%s\n' "${LIBURING_COMMIT}"
    printf 'tempo_patch_chain=v2->v3->v4\n'
    printf 'python_dependency=%s\n' "${PYTHON_DEPENDENCIES[@]}"
    printf 'tempo_v2_patch_sha256=%s\n' "$(sha256sum "${TEMPO_V2_PATCH}" | awk '{print $1}')"
    printf 'tempo_v3_patch_sha256=%s\n' "$(sha256sum "${TEMPO_V3_PATCH}" | awk '{print $1}')"
    printf 'tempo_v4_patch_sha256=%s\n' "$(sha256sum "${TEMPO_V4_PATCH}" | awk '{print $1}')"
    printf 'datastates_so_sha256=%s\n' "$(sha256sum "${DATASTATES_SO}" | awk '{print $1}')"
    printf 'datastates_core_so_sha256=%s\n' "$(sha256sum "${DATASTATES_CORE_SO}" | awk '{print $1}')"
    python -c 'import platform, torch; print(f"python={platform.python_version()}"); print(f"torch={torch.__version__}"); print(f"cuda={torch.version.cuda}")'
    # Avoid SIGPIPE under `set -o pipefail` when head exits after one line.
    "${CXX}" --version | sed -n '1p' | sed 's/^/compiler=/'
} > "${MANIFEST}"
echo "Baseline environment ready at ${RUNTIME}"
