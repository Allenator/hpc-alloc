# hpc-alloc

`hpc-alloc` submits ordinary Slurm batch jobs on Yale YCRC clusters and makes running allocation jobs reachable from a laptop over SSH. It is intended for interactive development on partitions where `salloc` is unavailable, as well as short-lived batch commands whose output should stream back to the caller.

hpc-alloc requires Python 3.11 or newer and keeps all state in its own configuration file and SQLite database.

## How it works

For `hpc-alloc up`, the client submits a sleeper with `sbatch`, waits for Slurm to assign a node, and writes an SSH alias such as `hpc-bouchet.dev`. The alias ProxyJumps through the cluster login host. Login and compute-node connections use OpenSSH multiplexing, so one Duo authentication can serve later polling, SSH, and rsync commands.

For `hpc-alloc run`, the client submits the requested command as a normal batch job. A foreground run follows the output and returns the batch exit status when final accounting provides it. If two exact successful queue observations establish finality before accounting supplies an exit status, it returns 0 for `COMPLETED` and 1 for every other final state. `--detach` leaves the job running for a later `hpc-alloc logs ... -f`.

Requirements:

- Python 3.11 or newer
- OpenSSH, `ssh-keygen`, and `rsync`
- an active Yale VPN connection and a YCRC account

## Install and setup

```bash
git clone <this-repository-url>
cd hpc-alloc
./install.sh
hpc-alloc setup --netid YOUR_NETID
```

`install.sh` verifies Python, the launcher, the complete bundled skill package, and the complete runtime-module manifest under an isolated Python startup before creating any links. It then links the executable into `~/.local/bin` and the bundled skill into every agent harness it detects: Claude Code at `~/.claude/skills/hpc-alloc` and Codex at `~/.codex/skills/hpc-alloc` (`$CODEX_HOME/skills/hpc-alloc` when set). Both harnesses read the same `skill/` package. Detection looks for the launcher on `PATH` or the harness home directory; pass `--claude` or `--codex` to install for a target regardless of detection, and note that a run detecting no harness at all fails rather than reporting a success that installed no skill. These are symlinks, so keep the checked-out repository at its installed path.

Codex sandboxes commands to the workspace with restricted network access by default, while hpc-alloc needs the network on every cluster-touching command and writes durable state under `~/.config/hpc-alloc/` and `~/.ssh/`. Approve those escalations when Codex requests them, but keep any standing approval to the read-only verbs — `status`, `config`, `avail`, `partitions`, `why`, and `logs`. A blanket `hpc-alloc` approval rule also pre-approves `run` and `ssh`, which execute arbitrary remote commands, along with `sync --delete`, `cancel`, and `down --all`; leave those to prompt each time. A denial that interrupts a submit is an unknown remote outcome, not proof that nothing was submitted, so run `hpc-alloc status` and reconcile any unresolved operation rather than retrying.

`setup` validates and writes the authoritative config, initializes the SQLite state database, finds or creates an SSH key, and adds one `Include` to `~/.ssh/config`. Upload the printed public key at <https://sshkeys.ycrc.yale.edu/>, wait for propagation, then authenticate:

```bash
hpc-alloc connect
```

Stateful commands hold a shared configuration-scope lock, while `setup` holds the exclusive side through its authoritative recheck and all mutations. A forced setup cannot change the NetID, remove a blocker-referenced cluster, or change that cluster's normalized login host while any job is non-final or any operation is unresolved. Same-scope replacement remains allowed. Use `hpc-alloc status`, finish or cancel remaining jobs, and run the printed `hpc-alloc recover OPERATION_ID` commands before changing scope. Once every job is final and every operation resolved, `setup --force` may change it.

`connect --push` performs the same bootstrap with one Duo push. Tell the user to expect and approve that push before invoking it from an unattended agent.

## Common workflow

```bash
# Inspect local journal entries and exact hpc-alloc jobs found on every cluster.
hpc-alloc status

# See available capacity, then create a persistent CPU allocation.
hpc-alloc avail
hpc-alloc up --name dev

# Allocate a GPU only for the duration of a command.
hpc-alloc run -p gpu_h200 -G h200:1 -c 8 --mem 64G \
  --chdir '~/project' -- python train.py

# Submit a long command without following it now.
hpc-alloc run --detach -G h200:1 -- python train.py
hpc-alloc logs bouchet:123456 -f

# Use and synchronize a running allocation.
hpc-alloc ssh bouchet:dev -- nvidia-smi
hpc-alloc sync bouchet:dev ./project '~/project'
hpc-alloc sync bouchet:dev '~/project/results' ./results --pull

# Diagnose or release work.
hpc-alloc why bouchet:dev
hpc-alloc cancel bouchet:123456
hpc-alloc down bouchet:dev
```

Use `up --dry-run` or `run --dry-run` to print a paste-ready submission command without connecting or changing local state. The command keeps the remote home as `${HOME:?}`, so execute it in the target login shell; relative paths and `~/...` working directories then resolve beneath that account's home directory. A command you paste and run yourself creates a real job that hpc-alloc neither journals nor tracks — its comment is tagged `dryrun-` so that `status` and recovery can tell it apart from a managed one.

## Commands

| Command | Purpose |
|---|---|
| `setup [--netid NETID] [--cluster NAME] [--host HOST] [--identity-file PATH] [--force]` | Create the config, state database, key material, and managed SSH include. Existing config requires `--force`. `--force` never re-keys: an `identity_file` already in the config is kept, because the key it names is the one registered with the cluster. Change it deliberately with `--identity-file`; if a configured key has vanished from disk, setup fails rather than silently substituting one that the cluster will reject. |
| `config [--cluster NAME] [--json]` | Validate config and show the effective resource values without contacting a cluster. |
| `connect [--cluster NAME] [--reset] [--push]` | Establish or heal the login master and health-check known allocation nodes. |
| `up [--name NAME] [--cluster NAME] [resources] [--idle-timeout MIN] [--no-wait] [--wait-timeout SEC]` | Submit a persistent sleeper allocation. The default waits for a node and exits 0 once one is ready; if the wait expires with the job still queued it exits 4, and the job stays submitted and tracked. `--no-wait` returns after durable submission acknowledgement without observing the scheduler state. |
| `run [--cluster NAME] [resources] [--chdir DIR] [--detach] -- CMD...` | Submit a command. Foreground mode follows output and returns the accounting exit status or the documented final-state fallback. |
| `status [--json]` | Reconcile locally journaled jobs and classify hpc-alloc-tagged queue rows across all configured clusters. |
| `why [TARGET] [--cluster NAME] [--json]` | Explain a queued, running, uncertain, or final job selected by name, job ID, or `@operation`. |
| `logs TARGET [--cluster NAME] [-n/--lines LINES] [-f/--follow]` | Read or follow a managed job log by convenience or durable selector. |
| `cancel (JOBID\|@OPERATION) [--cluster NAME]` | Cancel a managed job only after exact remote identity verification. |
| `down NAME\|JOBID\|@OPERATION\|--all [--cluster NAME]` | Cancel one or all managed allocation jobs. The target is required: `down` is irreversible, so it never guesses which allocation you meant. |
| `ssh [--cluster NAME] [NAME\|JOBID\|@OPERATION] [-- CMD...]` | Open an allocation shell or run a command there. |
| `sync (NAME\|JOBID\|@OPERATION) SRC DST [--cluster NAME] [--pull] [--delete]` | Transfer files with rsync through the allocation alias. rsync expands the remote path through the remote login shell — which is what makes `'~/project'` work — so the remote path is restricted to `A-Za-z0-9_@%+=:,./~-` and a path containing a space, quote, glob, or `$(...)` is rejected rather than silently re-split by that shell. |
| `avail [--cluster NAME] [-p PARTITION] [--json]` | Summarize idle CPUs and free GPUs for one cluster (idle GRES is not a schedulability guarantee), marking each partition with whether your account may submit to it (`ELIGIBLE` column; `eligible` in `--json`). With `--for` plus a request (`-G/-c/--mem/-t/-C`), probe where that request would start soonest across the eligible partitions — a scheduler dry-run that submits no job, ranked by advisory estimated start; preemptible or short pools are shown but marked, and `--json` carries a `capped` flag when more eligible partitions existed than were probed. |
| `partitions [--cluster NAME] [--json]` | Show live partition limits, GRES, and feature data for one cluster, plus whether your account may submit to each (`eligible`). |
| `recover [OPERATION_ID] [--cluster NAME] [--abandon] [--yes]` | Reconcile ambiguous submit/cancel operations by exact queue or accounting identity, or explicitly abandon one local intent. |

Resource flags shared by `up` and `run` are `--cluster`, `-p/--partition`, `-t/--time`, `-c/--cpus`, `--mem`, `-G/--gpus`, `-C/--constraint`, and `--dry-run`. `up` additionally accepts `--idle-timeout`, `--no-wait`, and `--wait-timeout`. Where a flag defaults, it does so quietly: `up --name` is `dev`, `up --wait-timeout` is 1800 seconds, `logs -n` is 100 lines, and `setup --cluster` is `bouchet` — the one `--cluster` anywhere with a default, since every other command resolves it from the config instead. `--idle-timeout` guards against a GPU allocation sitting idle, so it requires `-G/--gpus` and is rejected without it.

When you request a specific GPU type with `-G TYPE:N` but omit `-p/--partition`, `up` and `run` select the partition: if the default partition already offers that type it is kept, otherwise the tool auto-selects the single dedicated partition that offers it and prints which it chose; it refuses locally when several qualify, and when only preemptible or short pools offer the type it names them with `-p` guidance rather than dispatching an unschedulable request. Dedicated means a partition not matched by the cluster's `nondedicated_partition_globs` (default `scavenge*`, `*devel`). Selection reads a cached GPU-topology map, so a `--dry-run` resolves offline from a warm cache and prints the same partition a real submit would. It always prints a command, and warns on stderr whenever it could not resolve one: on a cold cache it says the topology is not cached, and on a warm cache that cannot pick — several dedicated partitions qualify, or none does — it gives the same refusal a real submit would. Either way it prints the configured default, so the printed partition is only what a real submit would use when no warning accompanies it. A live submit instead falls open to the configured default if the topology cannot be read.

Before dispatching, `up` and `run` refuse a partition your account, QOS, or groups PROVABLY cannot use — a best-effort accelerator that reads cached access rules (warmed by `connect`), so a clear access error is caught before the round-trips without a fetch. It falls open on any uncertainty (missing, empty, or partition-scoped access data), and the scheduler itself is the authoritative gate: a deterministic rejection is returned as a clean local failure (exit 1), not an ambiguous submission to recover.

Numeric Slurm durations support all six documented forms: `minutes`, `minutes:seconds`, `hours:minutes:seconds`, `days-hours`, `days-hours:minutes`, and `days-hours:minutes:seconds`. Every subfield that follows a colon must be exactly two digits from `00` through `59`. A field that no colon precedes is unbounded and needs no padding, so `5` is five minutes, `90:30` is ninety minutes and thirty seconds, and `100:00:00` is a hundred hours. Signs, whitespace, and symbolic values such as `INFINITE` or `UNLIMITED` are not accepted. Every all-zero spelling is also rejected because Slurm interprets a zero duration as requesting no time limit; specify an explicit, finite nonzero duration instead.

## Authoritative configuration

The only application config is `~/.config/hpc-alloc/config.toml`. Its schema is strict: unknown tables, unknown keys, invalid types, and invalid values are errors. See [`config.example.toml`](config.example.toml) for a complete example.

```toml
[identity]
netid = "abc123"

[ssh]
identity_file = "~/.ssh/id_ed25519"

[defaults]
cluster = "bouchet"
partition = "day"
gpu_partition = "gpu"
time = "4:00:00"
cpus = 2
idle_timeout = 30
# mem = "16G"   # no built-in default; omitted, Slurm applies cluster policy

[cluster.bouchet]
host = "bouchet.ycrc.yale.edu"
```

`[identity].netid` and at least one `[cluster.NAME].host` are required. The `[ssh]` and `[defaults]` tables may be empty. Every resource key accepted in `[defaults]` may also appear in a cluster table; a cluster value overrides the global default.

The values shown above are the built-in defaults, with one exception: `mem` has none. Set it only to impose a floor of your own — a value here applies to every job, GPU work included — and leave it out to let each cluster's policy decide.

`[defaults]` or a `[cluster.NAME]` table may also set `nondedicated_partition_globs`, a non-empty list of fnmatch globs marking partitions as non-dedicated (preemptible or short-lived); these are excluded from `-G TYPE:N` GPU auto-selection and flagged in `avail --for`. It defaults to `["scavenge*", "*devel"]` — set it when a cluster names those pools differently.

Cluster IP literals may be bare or enclosed in matching brackets. Brackets must enclose a valid IP address and are removed before generating SSH config.

Invocation precedence is:

```text
CLI flag > selected [cluster.NAME] value > [defaults] value > built-in fallback
```

If several clusters are configured, commands that must choose a cluster implicitly require `[defaults].cluster` (notably `status`) or `--cluster` where that flag is supported. A cluster-qualified job selector supplies its own cluster, while unfiltered `recover` and `down --all` may span clusters without a default. There is no fallback host convention in the config parser: each cluster table must declare its host explicitly.

## Durable job selectors

The operation ID is the durable identity; Slurm job IDs can be recycled and logical names can repeat. The canonical selector is:

```text
@operation_id
cluster:@operation_id
```

Commands that accept a managed job target also accept the convenience forms, qualified or bare:

```text
name
jobid
cluster:name
cluster:jobid
```

Examples are `bouchet:@08a3a68f1ad04ac595836695e0e9cc95`, `bouchet:dev`, and `bouchet:123456`. `why`, `logs`, `down`, `ssh`, and `sync` accept names, numeric job IDs, and operation selectors; `cancel` accepts only a numeric job ID or operation selector. Numeric and name selectors prefer one current non-final job over retained history. Other ambiguity errors list the operation selectors that remain individually addressable. A qualifier and `--cluster` must agree. Numeric IDs are temporary locators: when Slurm reuses one, the old operation is reconciled only through its exact operation-derived accounting identity and the replacement is reported separately. Jobs are never rebound by numeric ID.

An exact durable selector for a retained final record remains meaningful after a permitted setup removes its cluster. Locally answerable history such as a no-ID `SUBMIT_FAILED` or `ABANDONED` diagnosis remains accessible by the recorded `cluster:@operation` selector. Any action or history read that needs SSH, scheduler, accounting, or remote-log access still requires that recorded cluster to be present in the current configuration and never falls back to another cluster.

`status` polls every configured cluster. Failure of the primary cluster is an error. An unavailable secondary is reported on stderr, its local records are preserved as `UNCERTAIN`, and no absence-based cleanup is performed. Commands that act on one job use that job's cluster. `down --all` can span clusters; `--cluster` restricts it. A host-key change is an integrity failure on every cluster and always aborts status rather than degrading to `UNCERTAIN`.

## Durable state and recovery

Tool-owned state lives in `~/.config/hpc-alloc/state.db`, a mode-0600 SQLite database using WAL mode. SQLite may create `state.db-wal`, `state.db-shm`, and a rollback journal. Do not edit or copy individual files from an active database. This release has a fixed state schema and provides no migration tooling; archive or remove an older database only after accounting for any still-running remote jobs, then run setup again.

The database records the machine identity, jobs, lifecycle evidence and its final source, cluster caches, and a durable operation journal. Submission and cancellation follow a prepare/remote-call/acknowledge protocol with short local transactions:

1. Persist the intended mutation and its exact identity.
2. Leave the SQLite transaction before invoking SSH or Slurm.
3. Perform the remote mutation once.
4. Record the acknowledgement, or mark the operation ambiguous if the reply may have been lost.

Step four is why the tool exists. A lost reply is not a failure, it is an unknown — and the distinction is worth this much machinery because the two mutations are not symmetric. Replaying a submission can put a second GPU job on the cluster that nobody is watching; replaying a cancellation cannot do any harm. So an ambiguous mutation is never guessed safe to retry. `status` exposes it and prints the relevant operation ID. Reconcile it with:

```bash
hpc-alloc recover OPERATION_ID
```

Recovery is observation-only. It re-reads the queue and accounting under the job's exact identity and never issues another `sbatch` or `scancel`. It resolves the operation when the evidence is conclusive either way — the job is durably final, or it is demonstrably still alive and the cancellation never landed — and leaves it unresolved when the evidence cannot tell a landed mutation from a missed one. Unresolved is not a dead end: it means look again, not guess. `recover OPERATION_ID --abandon` discards the local intent alone, warns that a remote orphan may survive, and prompts unless `--yes` is given.

The exact rules live in one place: [the reference the bundled skill loads](skill/references/recovery-and-lifecycle.md). Which phase means what, which observations count as proof, why an empty accounting comment is accepted only alongside an exact job name, and why a requeue-eligible terminal needs two independent observations. That single copy is deliberate — a person and an agent acting on this contract must not be reading two versions of it, and when there were two, they disagreed.

### Exact ownership

Each mutation gets a random 32-hex-character operation ID. A Slurm job created by hpc-alloc uses both of these identifiers:

```text
job name: hpcalloc-v2-<alloc|run>-<operation_id>
comment:  hpc-alloc:v2:<owner_id>:<operation_id>:<host_label>:<kind>:<logical_name-or->
```

Live observations and mutation guards, including cancellation, require both the exact job name and complete comment. Terminal accounting and accounting recovery use the narrower omission rule above because `sacct` may return an empty comment. A matching numeric job ID, prefix, logical name, or machine host label alone is never sufficient. Host labels are safe deterministic display metadata; `owner_id` is authoritative. Jobs with malformed, legacy, or foreign live tags are not cancelled.

## Lifecycle and stream policy

Queue absence is evidence, not immediate proof that a job ended. A job missing from one poll may be draining, requeueing, or hidden behind a scheduler hiccup, so the lifecycle tracker grades what it sees — started, active, inactive, requeueing, terminal-candidate, final, uncertain — and requires confirmation before it calls anything final. Transport and scheduler failures never masquerade as job death, and final evidence is monotonic: a later weaker or missing observation cannot erase a terminal state or exit code already proved. The full taxonomy, the Slurm state mapping, and the evidence rules are in [the same reference](skill/references/recovery-and-lifecycle.md).

Foreground and follow behavior is intentionally command-specific. Every bullet below treats Ctrl-C, SIGTERM, and SIGHUP identically, so closing the terminal on a foreground `run` releases its job rather than leaking it for the whole walltime:

- Ctrl-C or SIGTERM while `up` is waiting does not cancel the acknowledged allocation. The CLI prints its canonical selector plus `status` and `down` guidance, then returns 130; the allocation may remain queued or running.
- Ctrl-C or SIGTERM during foreground `run` attempts to cancel the exact job, reports whether cancellation was confirmed, and returns 130. A cancellation that may have dispatched prints its operation ID and exact recovery command.
- A closed stdout pipe during foreground `run` likewise attempts exact cancellation and returns 141; inability to confirm cancellation is reported without replacing that status.
- An ordinary scheduler, transport, or log-read failure during foreground `run` does not cancel the job. The CLI prints the canonical selector and reattach/cancel guidance if it may continue, or logs/diagnosis guidance if durable finality was already reached, then propagates the original error.
- Ctrl-C or SIGTERM during `logs -f` detaches; the job continues and the CLI returns 130.
- A closed stdout pipe during `logs -f` detaches, leaves the job running, and returns 141.
- A clean `logs -f` returns 0 regardless of the job's final Slurm state.
- A foreground `run` returns the batch command's accounting exit code when available. Without an accounting exit code, it returns 0 for `COMPLETED` and 1 for another final state; it also forces a nonzero result for a non-`COMPLETED` state even if accounting reports status 0.

Progress and recovery notices go to stderr. JSON stdout remains machine-only. Argparse usage failures, including missing required arguments and invalid typed values, print usage and exit 2 before command dispatch; post-parse configuration, validation, scheduler, protocol, and other hpc-alloc application failures normally use exit 1. Exit 2 and exit 3 are contextual rather than globally reserved: a foreground batch command may itself return either status, while `ssh` and `sync` can return delegated OpenSSH or rsync statuses. Typed authentication, host-key, and transport failures use exit 3, and a possibly dispatched cancellation may surface its recovery guidance through a transport-class failure. Interpret these statuses together with the invoked command and stderr.

`up` uses exit 4 for "submitted, not ready yet": its wait expired without an active allocation on a visible node. Still queued is the usual reason and an ordinary outcome on a busy GPU partition, but every other non-final state at the deadline reports 4 as well — requeueing, suspended, or an observation too uncertain to classify — so read the printed state rather than assuming the queue. This is neither success nor failure. The job remains submitted, durable, and tracked, so it must not be resubmitted — wait for it with `status`, follow it with `logs -f`, or release it with `down`.

## JSON contracts

The JSON surfaces are intentionally explicit:

- `config --json` returns `config_file`, `state_file`, `primary_cluster`, the validated `config`, and `effective` resource values.
- `status --json` returns exactly three top-level arrays: `jobs`, `discovered`, and `operations`. Each `jobs` entry carries `selector`, `operation_id`, `jobid`, `cluster`, `name`, `kind`, `phase`, `scheduler_state`, `evidence_detail`, `ever_started`, `current_node`, `last_node`, `terminal_state`, `exit_code`, `final_source`, `partition`, `time`, `gpus`, and `alias`. A job finalized during reconciliation appears once in `jobs`, not again as a discovered conflict. Each `discovered` entry carries `cluster`, `jobid`, `operation_id`, `selector`, `job_kind` (`allocation` or `run`), `classification`, `state`, `node`, `partition`, `time_left`, `owner`, and `name`, where `classification` is one of `untracked-owned`, `other-machine`, `unresolved-operation-match`, `duplicate-operation`, `local-final-conflict`, or `operation-identity-conflict`. `operations` contains unresolved submit/cancel journal rows, each carrying `operation_id`, the target-job `selector`, `kind` (`submit` or `cancel`), `phase`, `cluster`, `target`, `jobid`, and `detail`; a `phase` is one of `PREPARED`, `ACKNOWLEDGED`, `AMBIGUOUS`, `CANCEL_PENDING`, `RESOLVED`, `FAILED`, or `ABANDONED`.
- `why --json` returns one `jobs`-shaped entry plus `status` (a duplicate of `phase`), `diagnosis`, and `detail`; a final job whose accounting record supplies them also carries display-only `elapsed` and `timelimit`.
- Enum-valued fields serialize in lowercase-hyphen form, not as the names used in prose: `final_source` is `accounting`, `confirmed-queue`, `submit-failed`, or `abandoned`. Match those exact strings.
- `avail --json` returns `{ "partitions": { ... } }`, where each partition object carries an `eligible` flag (true, false, or null when access data is unavailable); `avail --for --json` returns `{ "for": { resolved request }, "probes": [ { "partition", "preemptible", "schedulable", "start", "detail" } ], "capped": bool }`, ordered soonest-first, where `capped` is true when more eligible partitions existed than were probed. When the requested `-G TYPE:N` names a GPU type no partition offers, it instead carries an `error` string alongside empty `probes` and `capped` false.
- `partitions --json` returns an array of partition objects, each with an `eligible` flag (`true`, `false`, or `null` when access data is unavailable).

Consumers should use `kind` for managed jobs, `job_kind` plus `classification` for discovered rows, and canonical `selector` values for later actions. Do not parse display text.

## GPU policy

Prefer `hpc-alloc run -G TYPE:N -- ...` for GPU work so the GPU is held only while the command executes. A persistent GPU allocation created with `up -G` installs an idle watchdog; the built-in threshold is 30 consecutive minutes at at most 5% utilization. Configure `idle_timeout`, override it with `--idle-timeout MIN`, or explicitly pass `0` to disable the watchdog.

`hpc-alloc avail` reports current free capacity, and `partitions` supplies the live partition/GRES names. Request only the resources needed: larger requests usually wait longer and affect fair share.

## Files managed

- `~/.config/hpc-alloc/config.toml` — authoritative user configuration
- `~/.config/hpc-alloc/.config_scope.lock` — shared/exclusive configuration-scope serialization
- `~/.config/hpc-alloc/state.db*` — SQLite state, WAL, shared-memory, and rollback-journal sidecars
- `~/.config/hpc-alloc/operation-locks/*.lock` — retained per-operation advisory lock files; do not remove them while hpc-alloc processes may be running
- `~/.config/hpc-alloc/ssh_config` — lock-serialized regenerated login and allocation aliases
- `~/.config/hpc-alloc/.ssh_config.lock` — exclusive managed SSH projection serialization
- `~/.config/hpc-alloc/known_hosts` — managed host-key store, namespaced by cluster and physical node
- `~/.ssh/config` — only an `Include` for the managed SSH config is added
- `~/.ssh/id_ed25519_hpc_alloc` and `~/.ssh/id_ed25519_hpc_alloc.pub` — conditionally generated keypair when setup cannot reuse configured or standard key material
- `~/.ssh/hpc-alloc-*` — OpenSSH control sockets
- `~/.hpc-alloc/` on each cluster — operation-ID-named allocation and run logs

## Troubleshooting

- Interpret exit 3 in context. For an hpc-alloc authentication, host-key, or transport diagnostic, reconnect the VPN or run `hpc-alloc connect` as directed; for printed mutation ambiguity, run the exact `recover` command instead. A foreground batch command or delegated SSH/rsync process may independently return 3.
- If `status` lists an unresolved operation, run the exact `recover` command it prints. Do not submit or cancel again just because an SSH reply was lost.
- If a name or job ID is ambiguous, use the reported `cluster:@operation_id` selector.
- A host-key change is surfaced as a hard failure. Verify the new key with YCRC before changing known-hosts data.
- Run `hpc-alloc why TARGET` for pending reasons or final accounting evidence; optional queue diagnostics may be omitted during an ordinary transient failure without discarding the primary diagnosis.
- Walltime cannot be extended. Synchronize important work before the deadline.

## Tests

```bash
./tests/run.sh
```

The suite is offline and uses strict Python test doubles and framed-protocol fixtures. It covers configuration validation, SQLite permissions and transactions, concurrent mutation invariants, exact ownership, ambiguous submit/cancel recovery, SSH/Slurm failure classification, lifecycle traces, requeue-aware streaming, multi-cluster selectors, status identity-graph behavior, tracked-only package delivery, and documentation formatting. Release validation must use a Git-index or committed clean-tree export; green tests in a dirty workspace do not prove that newly created package files are included. The suite does not require a cluster, VPN, credentials, or network access.

## License

MIT — see [LICENSE](LICENSE).
