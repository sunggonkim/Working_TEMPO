# NERSC coding-agent safety warning

This project is developed on NERSC shared systems. Read and follow the
[NERSC AI coding-agent guidance](https://docs.nersc.gov/development/coding-agents/)
before running an agent or a diagnostic command.

## Required operating rules

- Never recursively traverse system or shared pseudo-filesystems such as
  `/`, `/proc`, `/sys`, `/usr`, `/global`, or `/cfs`. Do not use `find`, `bfs`,
  `du`, `tree`, `ls -R`, or an unscoped recursive `grep`/`rg` there.
- Inspect only explicitly named files or narrowly scoped paths inside this
  repository. Do not enumerate Lustre OSC trees, PCI devices, or all network
  interfaces; use a known path supplied by the experiment contract instead.
- Do not submit, cancel, or retry Slurm work without explicit user approval.
  A pending job must be observed with one monitor, not by repeated logins or
  multiple watchers.
- When the user explicitly approves a long-lived interactive allocation,
  prefer `salloc --no-shell` and launch compute work with an explicit
  `srun --jobid=<allocation>` attachment.  Verify the exact RUNNING job,
  node/GPU/time/QOS receipt and that no GPU step is active before launching
  one new step.  This keeps an SSH/chat PTY SIGHUP from releasing the whole
  allocation; it does not authorize duplicate allocations or background
  watchers.
- Do not run persistent polling, login loops, compute-node SSH, or background
  agents. Prefer scheduler mail notifications and a single, infrequent status
  check.
- Keep diagnostics read-only and bounded. Record the exact command and scope
  in the experiment log; stop immediately if a command would scan outside the
  project directory.

The user remains responsible for commands, data access, resource usage, and
external activity performed by an AI agent. When in doubt, stop and ask before
accessing a shared system path or launching a job.
