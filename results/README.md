# Local experiment artifacts

Only four historical G1 inputs needed for C0 offline replay are retained
locally in `c0_replay_local/`. All other raw jobs, logs, snapshots, and copied
source trees were removed. The directory is intentionally ignored by Git.

Claim status and compact summaries live under `paper/`, particularly
`STATUS.md`, `TEMPO_RD_COMPACT_EVIDENCE.json`, and
`TEMPO_RD_SCHEDULER_STOP_DECISION.md`.

Do not reinterpret a local result directory as a promoted claim unless it
is reflected in status and compact evidence. These replay inputs are not a
positive scheduler result.
