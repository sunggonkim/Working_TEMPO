# TEMPO LMCache source patch

The parent repository pins upstream LMCache commit
`227d13f5c9fdb52ddb933641d34331f678de03a0`. The actual TEMPO experiments also
use local LMCache integration changes that cannot be represented by a dirty
submodule pointer. They are exported in
[`lmcache_tempo_current.patch`](lmcache_tempo_current.patch).

- patch SHA-256:
  `fa34d505ac72ed199a1a179fcd0e14f2d605d03b698edf5518cf5a8c13e4f76d`
- base commit: `227d13f5c9fdb52ddb933641d34331f678de03a0`
- inventory: 8 modified upstream files and 2 TEMPO NIXL hotpath files
- delta: 758 insertions, 48 deletions

Apply from the LMCache submodule root:

```bash
git apply --check ../../eval/sota_4node/lmcache_tempo_current.patch
git apply ../../eval/sota_4node/lmcache_tempo_current.patch
```

The patch was checked against a detached clean worktree at the pinned base
commit. This parent-repository patch is the reproducible source artifact; the
dirty state of a local submodule is not evidence and is not required after the
patch is applied.

The changes cover proxy cache-read control, scheduler-role completion handling,
cache-engine and asynchronous PD lifecycle accounting, NIXL channel behavior,
the TEMPO hotpath helpers, and focused LMCache tests. They do not modify system
network configuration, require root, or bypass the Perlmutter execution rules.
