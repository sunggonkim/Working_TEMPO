# Repository agent safety

Before operating on Perlmutter, read [NERSC_AGENT_SAFETY.md](NERSC_AGENT_SAFETY.md).
Those rules are mandatory for this repository: no recursive traversal of
system/shared filesystems, no unscoped searches, no background watchers, and
no Slurm submission or retry without explicit user approval.
