# hpc-alloc

Allocate Yale YCRC compute nodes (Bouchet) for interactive development from your laptop — including on partitions where `salloc` is not allowed.

## Why this exists

On YCRC's public clusters, interactive jobs (`salloc`) are only permitted on the `devel`/`gpu_devel` partitions, which are capped at 6 hours, 4 CPUs, and 2 GPUs per user. If you want to develop against a real H200/B200 node, or hold a node for a full day, you need the batch partitions — but those don't allow interactive jobs.

`hpc-alloc` sidesteps this with a standard, scheduler-friendly pattern:

1. Submit a **sleeper batch job** (`sbatch --wrap 'sleep infinity'`) with your resource requirements — legal on any partition.
2. Wait for it to start and learn which node it landed on.
3. Write an SSH alias (e.g. `bouchet-dev`) that **ProxyJumps through the login node** to the compute node. YCRC permits SSH into nodes where you have a running job, and [documents this tunneling pattern](https://docs.ycrc.yale.edu/clusters-at-yale/access/advanced-config/).

Both hops use SSH multiplexing (`ControlMaster`/`ControlPersist 4h`): you answer Duo MFA once on the login connection, and the first command to a compute node establishes a persistent master there too — after that, every command (squeue polling, remote execution, rsync) is a ~50ms channel-open on an existing stream instead of a fresh handshake. The compute-node alias is plain SSH, so anything that speaks SSH works: `ssh bouchet-dev`, VS Code Remote-SSH, rsync, Claude Code.

Requirements: an active Yale VPN connection and a YCRC account. No pre-existing SSH setup needed — `hpc-alloc setup` creates everything.

## Install

```bash
git clone <this repo> && cd hpc-alloc
./install.sh    # symlinks the CLI into ~/.local/bin and the skill into ~/.claude/skills
```

## One-time setup

```bash
hpc-alloc setup --netid YOUR_NETID
```

This finds (or generates) an SSH key, writes a managed SSH config (`~/.config/hpc-alloc/ssh_config`, included from `~/.ssh/config`), and prints your public key. Upload the key at <https://sshkeys.ycrc.yale.edu/> (VPN required, takes a few minutes to propagate), then verify:

```bash
hpc-alloc connect    # Duo MFA prompt happens here, once per ~4h
```

## Usage

```bash
# CPU dev node on the default 'day' partition (4h, 2 CPUs, 10G)
hpc-alloc up

# Check free capacity first: idle CPUs and free GPUs by type, per partition
hpc-alloc avail

# Run a GPU command as a short-lived batch job — output streams back, Ctrl-C cancels,
# exit code mirrors the job. GPUs are only held while the command runs (see GPU policy below).
hpc-alloc run -p gpu_h200 -G h200:1 -c 8 --mem 64G --chdir '~/project' -- python train.py

# Long jobs: submit detached, follow (or reattach) later, cancel by id if needed
hpc-alloc run --detach -p gpu_h200 -G h200:1 -- python train.py
hpc-alloc logs 123456 -f
hpc-alloc cancel 123456

# Or hold an H200 node for 8 hours (self-releases after 30 idle GPU minutes by default)
hpc-alloc up --name h200 -p gpu_h200 -G h200:1 -c 8 --mem 64G -t 8:00:00 --idle-timeout 90

# See what you have: allocations AND run jobs, live GPU utilization, expiry flags
hpc-alloc status

# Job stuck or dead? One-shot diagnosis (pending reason, quota caps, maintenance
# windows, walltime/OOM/cancellation forensics + log tail)
hpc-alloc why h200

# Use it
ssh bouchet-h200                                   # plain SSH / VS Code Remote-SSH host
hpc-alloc ssh h200 -- nvidia-smi                   # one-off remote command
hpc-alloc sync h200 ./myproject '~/myproject'      # push code (rsync, incremental)
hpc-alloc sync h200 '~/myproject/out' ./out --pull # pull results

# Release it
hpc-alloc down h200
```

`hpc-alloc up --dry-run ...` prints the exact `sbatch` command without connecting. `hpc-alloc partitions` shows live partition limits, GPU GRES names, and `-C` feature tags. `status`, `avail`, `why`, and `partitions` all take `--json` for machine consumption. `hpc-alloc --help` includes a Bouchet partition cheatsheet.

## Configuration

Optional TOML config at `~/.config/hpc-alloc/config.toml` — `hpc-alloc setup` creates it (pinning your SSH key), or copy [`config.example.toml`](config.example.toml). Precedence: **CLI flags > config file > built-in defaults.**

- `[defaults]` — `cluster`, `partition`, `gpu_partition`, `time`, `cpus`, `mem`, `idle_timeout`: what you usually want, so you stop retyping flags.
- `[ssh] identity_file` — pins the key with `IdentityFile` + `IdentitiesOnly` in the generated SSH config. Without this, ssh offers every key it knows and the server can hit "Too many authentication failures" before trying the right one.
- `[cluster.<name>]` — per-cluster `host` and partition overrides, for clusters whose partition layout differs from Bouchet's.

The division of labor: config.toml = who you are and what you usually want (yours to edit); `state.json` = what currently exists (tool-owned); flags = this invocation's exceptions. `hpc-alloc config` prints the effective merged defaults with provenance (`--json` for agents), and every submission echoes the actual partition/walltime/idle-timeout it used. Parsing uses `tomllib` on Python ≥ 3.11 and falls back to a built-in subset parser (strings/ints/bools, dotted sections) on older interpreters.

## Bouchet quick reference

| Partition | Max time | Notes |
|---|---|---|
| `day` (default) | 1 day | 1000 CPUs/user |
| `week` | 7 days | 96 CPUs/user |
| `gpu` | 2 days | 8 GPUs/user; RTX 5000 Ada, L40S, … |
| `gpu_rtx6000` / `gpu_h200` / `gpu_b200` | 2 days | 16 GPUs/user |
| `bigmem` | 1 day | up to 8 TB/user |
| `devel` / `gpu_devel` | 6 h | the only public partitions where salloc works |
| `scavenge` / `scavenge_gpu` | 1 day | preemptable |

GPU syntax: `--gpus TYPE:N`, e.g. `h200:1`, `rtx_5000_ada:2`. Defaults if unspecified: 1 node, 1 task, 5120 MB RAM per CPU.

## GPU idle policy

YCRC runs a daemon that detects jobs with idle GPUs: you get a warning email first, and if the GPUs stay unused the job is cancelled ([policy](https://docs.ycrc.yale.edu/clusters-at-yale/policies/)). A sleeper allocation holding an unused GPU is exactly what it targets, so hpc-alloc works with the policy rather than around it:

- **`hpc-alloc run`** executes commands as short-lived batch jobs — GPUs are allocated only while the command actually runs, so there is never anything to flag. Prefer it for GPU work; keep your persistent dev seat on a CPU partition.
- **Direct GPU allocations self-release**: `up -G ...` installs a watchdog in the sleeper job that exits after 30 consecutive minutes of ≤5% GPU utilization — the same threshold at which YCRC sends its warning (`--idle-timeout MIN` to tune, `0` to disable). You release on your own terms instead of collecting warnings, and `hpc-alloc up` re-creates the node in one command.
- `hpc-alloc status` shows live GPU utilization per allocation, so you can see what the daemon sees, and `hpc-alloc avail` shows free GPUs by type before you request any.

## Connection resilience

VPN drops, laptop sleep, and network changes are detected and healed rather than misreported:

- Every operation first verifies the connection end-to-end (a multiplexed master can look alive while its TCP is dead — `-O check` alone can't tell). Stale masters are torn down and re-established automatically; orphaned control sockets are swept, since a dead socket file would otherwise make OpenSSH silently disable multiplexing and re-prompt Duo on every command. Healing is evidence-based: a failed command triggers a probe first, and masters are only torn down when the probe fails — a slow squeue or a cluster-side hang never kills the SSH sessions your editor is riding on.
- Transport failures and scheduler failures are kept apart, because their remedies are disjoint: when slurmctld itself is erroring over a healthy connection, polling commands retry briefly and then exit 1 with a message saying so — they never enter the reconnect/Duo path (which can't help) and never loop forever.
- A network failure is never mistaken for job death: allocations are only removed from state when the queue was actually consulted and the job is confirmed gone. Sleeper jobs and `run` jobs keep running on the cluster through any client-side outage.
- Interactive commands ride out drops (during `up`'s wait, it retries for up to 10 minutes and re-prompts Duo when the network returns). Non-interactive callers get **exit code 3** with instructions, because re-auth needs your second factor: reconnect the VPN, run `hpc-alloc connect`, retry.
- **Terminal-free re-auth**: `hpc-alloc connect --push` authenticates via Duo push — ssh re-invokes hpc-alloc itself as its `SSH_ASKPASS` program, which answers Duo's menu with "1" (send push), and you approve on your phone. No helper file is written and no secret passes through the client, so agents can run it (after telling you to expect the push). The pinned key must be usable non-interactively: passphrase-protected keys are detected up front with an `ssh-add` hint instead of a timeout blaming the push.
- After a wake-from-sleep, note that walltime kept counting — the allocation may have expired. `hpc-alloc connect` health-checks every connection and says so; `hpc-alloc status` cleans up.

## Claude Code skill

`install.sh` links `skill/` to `~/.claude/skills/hpc-alloc`, making the skill available in every project. Claude then knows how to allocate nodes sized to the task, sync code, run commands remotely (with `module load`, tmux for long jobs), and release allocations — while leaving interactive steps (VPN, Duo via `hpc-alloc connect`) to you.

## Troubleshooting

- **Exit code 3 / "connection lost"** — reconnect the VPN if needed, run `hpc-alloc connect` in a terminal (Duo prompt happens there; it also health-checks every node connection), then retry. Nothing was lost: jobs keep running and allocations stay in state.
- **"could not connect" during setup/connect** — VPN not up, key not yet propagated (wait a few minutes after uploading), or wrong NetID.
- **Job stuck PENDING** — run `hpc-alloc why <name>`: it distinguishes normal contention (with the scheduler's start estimate and priority breakdown) from per-user quota caps (waiting won't help until your jobs finish) and maintenance reservations (a shorter `-t` may start immediately). Then `--no-wait`, fewer resources, or `scavenge`(-`_gpu`) if you can tolerate preemption.
- **"allocation has ended"** — `hpc-alloc why <jobid>` says whether it was walltime, OOM, the idle watchdog, or an admin cancellation, with a log tail. `hpc-alloc status` cleans up; `hpc-alloc up` re-creates it.
- **Node unreachable but job RUNNING** — `hpc-alloc connect` probes and heals each node connection; then try `ssh -v <alias>`.
- **A hung interactive `ssh` session** — type `~.` (tilde, dot) at the start of a line to force-close it, then `hpc-alloc connect`.
- **"HOST KEY VERIFICATION FAILED"** — the server's key changed (login-node reimage, or worst case interception). The tool surfaces ssh's warning and refuses to proceed; verify with YCRC, then `ssh-keygen -R <hostname>`.
- **An UNTRACKED-allocation entry in `status`** — a running hpc-alloc job this machine doesn't track. Jobs are ownership-tagged (`--comment=hpc-alloc:<id>:<hostname>`, where the id is a random per-machine identity persisted in state.json — hostname changes and DHCP renames can't disown your jobs), so allocations created from your *other* machine are labelled "created on '<machine>' — manage it there" and never get a cancel hint; a genuine orphan (lost record on this machine) shows the `hpc-alloc cancel <jobid>` hint. Entries marked "just submitted" are an `up` from another window — leave them alone.
- **Exit codes and streams**: `run` mirrors the job's exit code (non-COMPLETED ends are nonzero; it refuses to claim success without a final accounting record); `logs -f` always exits 0 after a clean stream; connection loss is exit 3; Slurm/scheduler failures are exit 1 (the message says reconnecting won't help — don't loop on `connect`); a closed output pipe is exit 141. All progress notices go to stderr — stdout of `--json` commands is pure JSON.
- **Walltime cannot be extended.** Slurm won't let users raise a running job's time limit; sync your work out and re-allocate.

## Files it manages

- `~/.config/hpc-alloc/state.json` — NetID, machine identity, clusters, live allocations.
- `~/.config/hpc-alloc/ssh_config` — regenerated from state; included from `~/.ssh/config` via one `Include` line (the only edit ever made to your own config).
- `~/.ssh/hpc-alloc-*` — ControlMaster sockets.

Everything on the cluster side is a normal batch job named `hpcalloc-<name>` plus logs in `~/.hpc-alloc/` (pruned after 30 days); `scancel` works on it like any job.

## Tests

`tests/run.sh` runs the whole suite offline — unit tests for the parsers plus end-to-end scenarios (connection loss, slurmctld errors, job death, GPU status, cancel safety, capacity digests) against a fake `ssh` shim in `tests/shim/`. No cluster, network, or credentials needed.
