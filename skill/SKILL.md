---
name: hpc-alloc
description: Allocate and use Yale YCRC compute nodes for development, submit CPU/GPU batch commands, synchronize files, diagnose jobs, reconcile ambiguous mutations, and safely release exact hpc-alloc-owned jobs. Use when a user asks to inspect, allocate, use, recover, or release YCRC cluster resources.
---

# hpc-alloc

Use `hpc-alloc` from the user's laptop to manage ordinary Slurm batch jobs and SSH access to running allocation jobs. It requires Python 3.11+ and owns only the jobs it created, identified by the exact durable operation identity it stamped on them; never infer ownership from a configuration file, `state.json`, job name, comment, or alias.

## Start safely

1. Run `hpc-alloc status --json` before allocating or acting on existing work.
2. Reconcile every unresolved operation with its printed `hpc-alloc recover OPERATION_ID` command before retrying a mutation.
3. Run `hpc-alloc config --json` and, when resource choice matters, `hpc-alloc avail --json` or `hpc-alloc partitions --json`.
4. Reuse a suitable active allocation; otherwise create a development seat with `up` or submit finite work with `run`.
5. Prefer the printed `cluster:@operation_id` selector for reattachment, history, and automation.
6. Ask before `cancel`, `down`, or recovery abandonment unless releasing that exact work is already part of the user's instruction.

Read [command contracts](references/command-contracts.md) before composing commands or interpreting selectors, JSON, exit statuses, stream behavior, or historical access.

Read [recovery and lifecycle](references/recovery-and-lifecycle.md) before changing durable state, using `setup --force`, interpreting lifecycle evidence, cancelling work, or recovering an operation.

## Establish connectivity

Require the user to establish the Yale VPN. For one-time setup, run `hpc-alloc setup --netid NETID`, have the user upload the printed public key at <https://sshkeys.ycrc.yale.edu/>, and run `hpc-alloc connect` in a terminal.

Before invoking `hpc-alloc connect --push`, tell the user to expect a Duo push and send at most one expected push.

Interpret exit 3 from stderr and command context rather than assuming the VPN is down: typed authentication, host-key, or transport failures use 3, but a batch command, delegated SSH, or rsync may also return 3, and an ambiguous mutation may require `recover`. Do not blind-retry.

Treat a host-key change as a hard stop until the user verifies the new key through a trusted YCRC channel.

## Apply non-negotiable safety rules

- Never run heavy computation on a login node.
- Never infer ownership from a logical name, numeric job ID, prefix, hostname, or a tag that merely looks like hpc-alloc's; require the exact durable operation identity.
- Never call `scancel` directly; use `cancel` or `down` so the CLI verifies identity and journals the mutation.
- Never retry an ambiguous submit or cancellation; use observation-only recovery and preserve the printed operation ID.
- Never edit or query `~/.config/hpc-alloc/state.db` to bypass the CLI, and never delete or copy one live SQLite sidecar in isolation.
- Never infer finality from one queue absence, one scheduler-terminal row, a transport failure, or an unavailable secondary cluster.
- Never cancel a `discovered` or `other-machine` row merely because it resembles an owned job; treat it as evidence and obtain explicit direction.
- Assume an ordinary read or follow failure leaves remote work unchanged; treat a transport loss around submit or cancellation as an unknown remote outcome requiring the journal workflow.
- Preserve walltime-sensitive outputs before the hard deadline; Slurm walltime cannot be extended.

## Use resources deliberately

Prefer `hpc-alloc run -G TYPE:N -- ...` for finite GPU work so the GPU is held only while the command runs. Use a CPU `up` allocation for editing and builds when practical.

Inspect live partition and GRES names instead of guessing. Request only the CPUs, memory, GPUs, and walltime required; shorter realistic requests can improve backfill and larger requests usually wait longer and consume more fair share.

Do not disable a persistent GPU allocation's idle watchdog with `--idle-timeout 0` without explicit user approval, and never create artificial GPU load to evade cluster policy.

Use `sync` to preserve important results, and load required environment modules inside each remote `ssh` or `run` command because compute-node sessions begin with a minimal environment.
