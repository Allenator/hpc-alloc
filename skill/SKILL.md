---
name: hpc-alloc
description: Allocate and use Yale YCRC compute nodes (Bouchet) for development — request a node with specific CPUs/GPUs/memory/time on any partition (not just devel), SSH into it, sync code, run commands remotely, run GPU jobs, and release it. Use when the user asks to allocate/get a cluster or GPU node, run or test something on the cluster, sync code to the cluster, or check/release cluster allocations.
---

# hpc-alloc — YCRC compute nodes for development

`hpc-alloc` allocates a Bouchet compute node by submitting a sleeper batch job (`sbatch --wrap 'sleep infinity'`) — this works on **every** partition, unlike `salloc`, which public clusters only allow on `devel`/`gpu_devel`. Once the job starts, the CLI writes an SSH alias (e.g. `bouchet-dev`) that ProxyJumps through the login node to the compute node. Both connections are multiplexed: Duo MFA is prompted at most once per ~4h, and repeated `hpc-alloc ssh <name> -- CMD` calls are near-instant (no per-command handshake), so prefer many small commands over one giant script.

**Exit code 3 means the connection to the cluster was lost and re-auth needs the human's second factor**: do not blind-retry. Either tell the user to run `hpc-alloc connect` in their terminal, or — after telling them to expect a Duo push on their phone — run `hpc-alloc connect --push` yourself and let them approve it. Never send pushes repeatedly without the user's awareness (push fatigue). Jobs and allocations survive connection drops; state is never lost to a network blip. A Slurm/scheduler failure over a healthy connection is exit 1 (the message says so) — `connect` cannot fix it; report it and retry later instead of entering the reconnect protocol.

## Prerequisites

- Yale VPN connected (the user must do this; you cannot).
- One-time `hpc-alloc setup --netid NETID` done, with the SSH key uploaded at https://sshkeys.ycrc.yale.edu/. If a command fails with "not configured yet", walk the user through setup.

## Commands

| Command | Purpose |
|---|---|
| `hpc-alloc status --json` | Allocations + a `runs` list of every untracked hpc-alloc job, each with `kind`: `run` (batch job, ends on its own), `recent` (just submitted — probably another window; leave alone), `other-machine` (owned by the user's other computer, `owner` names it — leave alone), or `orphan` (a sleeper this machine lost track of — holds resources until cancelled). **Always run this first.** Progress notices go to stderr; stdout is pure JSON. |
| `hpc-alloc config --json` | Effective defaults: the user's config.toml merged over built-ins, with provenance. No cluster contact. |
| `hpc-alloc avail [--json] [-p PART]` | Free capacity digest: idle CPUs and free GPUs by type per partition. **Run before requesting GPUs.** |
| `hpc-alloc up [--name N] [-p PART] [-t TIME] [-c CPUS] [--mem M] [-G GPUS] [--idle-timeout MIN]` | Allocate a dev node; blocks until it starts (`--no-wait` for busy partitions). |
| `hpc-alloc run [resources] [--chdir DIR] [--detach] -- CMD...` | Run CMD as a batch job. Foreground streams output and mirrors the exit code (non-COMPLETED ends are nonzero); Ctrl-C cancels the job. One arg after `--` = shell string; several = exact argv. **Preferred for GPU work.** |
| `hpc-alloc logs JOBID\|NAME [-f]` | Show (or follow, `-f`) a job's log — reattach to detached/interrupted runs. Ctrl-C/stop detaches and **never cancels** the job. |
| `hpc-alloc why [NAME\|JOBID] [--json]` | Diagnose a job: why pending (contention vs. quota cap vs. maintenance), how it's doing, or why it died (walltime/OOM/cancelled). **Use whenever a job is stuck or gone.** |
| `hpc-alloc ssh [name] -- CMD...` | Run a command on an allocated node (non-interactive, safe for you to call). Interactive shell (no CMD) is for the user only. |
| `hpc-alloc sync NAME SRC DST [--pull] [--delete]` | rsync local→node (or node→local with `--pull`). |
| `hpc-alloc cancel JOBID` | Cancel an hpc-alloc job by id (refuses jobs it didn't create). |
| `hpc-alloc down [name\|--all]` | Cancel allocation(s) and remove SSH aliases. |
| `hpc-alloc partitions [--json]` | Partition list with limits, GPU GRES names, and `-C` feature tags. |
| `hpc-alloc connect [--reset] [--push]` | (Re)establish + health-check all connections. `--push` authenticates via Duo push — agent-runnable: tell the user to approve on their phone. Without `--push`, user-run only (terminal Duo prompt). |

For rare queries with no subcommand, run raw Slurm commands over the login alias: `ssh bouchet-login -- 'sinfo ...'`.

## Defaults discipline

The user's `~/.config/hpc-alloc/config.toml` may change any default (cluster, partition, time, cpus, mem, idle-timeout), so this document never tells you what a default *is* — only what the built-in fallback would be.

1. **Pass only the flags the task actually requires** (e.g. `-G h200:1` for GPU work, `--mem` for a memory-hungry job); leave the rest unset so the user's configured preferences apply. Don't copy fully-flagged examples verbatim.
2. **Need to know an effective value before acting** (e.g. to tell the user what will happen)? Run `hpc-alloc config --json`.
3. **Trust the echo over any prediction**: `up` and `run` print the actual partition, walltime, and idle-timeout used at submission — that output is the ground truth for what happened.

Cluster-side facts in this skill — partition time limits, the ~30-min idle-GPU warning policy, the scratch purge, exit-code meanings, job-name prefixes — are invariants that no config changes.

## GPU etiquette (important)

YCRC monitors GPU jobs and **cancels ones whose GPUs sit idle** (warning email at ~30 min; repeat offenses risk account suspension). Never try to defeat this with fake GPU load. Instead:

- **Two-tier pattern (default for GPU work):** keep the persistent dev allocation on a CPU partition (`hpc-alloc up -p day`) for editing/builds, and execute GPU work with `hpc-alloc run -G h200:1 -- ...` so GPUs are held only while computing.
- Check `hpc-alloc avail` first; if the GPU type you want shows 0 free, prefer `run` (it queues) over holding a node, or pick another type/partition.
- Direct GPU allocations (`up -G ...`) self-release after a period of GPU idleness (built-in default 30 min, matching YCRC's warning threshold; the user's config may change it). `--idle-timeout MIN` overrides per-allocation; `0` disables — only with the user's explicit OK. Warn the user when `status` shows a GPU allocation with low `gpu_util`.
- For quick interactive GPU debugging, `-p gpu_devel` is the intended partition (6h, 2 GPUs).

## Choosing resources (Bouchet)

- CPU work: `day` (max 1 day; the built-in default partition) or `week` (max 7 days, 96 CPUs/user).
- GPU work: `-G TYPE:N` alone picks the configured GPU partition (built-in: `gpu`). Verified GRES names: `h200`, `b200`, `rtx_5000_ada` (on `gpu`), `rtx_pro_6000_blackwell` (on `gpu_rtx6000`), `l40s` (scavenge only). E.g. `-p gpu_h200 -G h200:1`. GPU partitions max 2 days (QoS-enforced; `sinfo` shows "infinite").
- Big memory: `-p bigmem --mem 512G` (max 1 day). Memory defaults to 5120MB per CPU — set `--mem` explicitly for memory-hungry work.
- Verify GPU GRES names and `-C` feature tags with `hpc-alloc partitions`; shorter `-t` improves queue position (backfill), and for `run` over-requesting time costs only queue position, never held resources.
- Request modestly (a dev node, not a production run): more resources = longer queue wait.

## Typical workflow

1. `hpc-alloc status --json` — reuse a live allocation if its resources fit; don't stack duplicates. Warn the user about anything `expiring_soon`.
2. Dev seat: `hpc-alloc up --name dev`, adding flags only for needs beyond the user's defaults (e.g. `-t 8:00:00` for a long session, `-c 8` for parallel builds). If PENDING for long, run `hpc-alloc why dev`, then consider `--no-wait` or a different partition.
3. Push code: `hpc-alloc sync dev ./project '~/project'` (incremental; re-run after edits).
4. Build/test on the dev node: `hpc-alloc ssh dev -- 'cd ~/project && make test'`. Load software with `module load ...` (e.g. `module load miniconda`) inside the command — nodes start with a bare environment.
5. GPU execution: check `hpc-alloc avail`, then `hpc-alloc run -p gpu_h200 -G h200:1 -c 8 --mem 64G --chdir '~/project' -- python train.py`. For long jobs use `--detach` (or a backgrounded foreground run) and follow with `hpc-alloc logs <jobid> -f`; the job survives client death either way.
6. Pull results: `hpc-alloc sync dev '~/project/results' ./results --pull`.
7. When the user is done for the day: ask before `hpc-alloc down` — idle allocations waste fairshare, but the user may want to keep the node.

## Rules

- Never run computation on the login node; everything heavy goes through `hpc-alloc ssh`/`run`.
- Never call `scancel` directly — use `hpc-alloc cancel JOBID` / `hpc-alloc down NAME`, which refuse jobs hpc-alloc didn't create.
- Walltime is a hard deadline and cannot be extended — sync results out before it hits (`status` flags `expiring_soon`).
- Cluster `scratch` storage purges files after 60 days; keep anything important in `project` or home.
- On exit code 3: stop; either ask the user to run `hpc-alloc connect`, or announce and run `hpc-alloc connect --push` (they approve on their phone), then resume. On a stuck or dead job: `hpc-alloc why` first, then act on its diagnosis.
- A `runs` entry with `"orphan": true` is a sleeper this machine created but lost track of (jobs are ownership-tagged via their comment with a persisted per-machine id). Report it and run `hpc-alloc cancel <jobid>` only with the user's explicit confirmation; never touch `kind: recent` or `kind: other-machine` entries.
- `hpc-alloc run` mirrors the job's exit code (any non-COMPLETED end is nonzero); `hpc-alloc logs -f` is a pure watcher and exits 0 after a clean stream regardless of the job's outcome.
- Multi-cluster: config may define several `[cluster.X]` entries, but only Bouchet is live-validated (multi-cluster paths are tested offline only). Secondary-cluster failures degrade softly with a note naming the remedy: `unreachable — skipping it` (network), `needs re-auth — run hpc-alloc connect --cluster <x>` (run it after announcing, or relay it), or a `HOST KEY VERIFICATION FAILED` warning (stop and tell the user — possible interception). Its allocations show as `UNKNOWN`; report the note rather than treating it as an error.
