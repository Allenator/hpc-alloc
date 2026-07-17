# hpc-alloc

`hpc-alloc` submits ordinary Slurm batch jobs on Yale YCRC clusters and makes running allocation jobs reachable from a laptop over SSH. It is intended for interactive development on partitions where `salloc` is unavailable, as well as short-lived batch commands whose output should stream back to the caller.

It keeps all of its state in its own configuration file and SQLite database, and owns only the jobs it created.

## How it works

For `hpc-alloc up`, the client submits a sleeper with `sbatch`, waits for Slurm to assign a node, and writes an SSH alias such as `hpc-bouchet.dev`. The alias ProxyJumps through the cluster login host. Login and compute-node connections use OpenSSH multiplexing, so one Duo authentication can serve later polling, SSH, and rsync commands.

For `hpc-alloc run`, the client submits the requested command as a normal batch job. A foreground run follows its output and exits with the batch command's own status, or with one derived from the job's final state when accounting supplies none; `--detach` returns immediately and leaves the job running for a later `hpc-alloc logs ... -f`.

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

`install.sh` validates everything before it links anything: Python, the launcher, the bundled skill package, and the runtime-module manifest, all under an isolated Python startup. It then creates symlinks — so keep the checked-out repository where it is — for:

- the launcher, at `~/.local/bin/hpc-alloc`
- the skill, in every agent harness it finds: Claude Code at `~/.claude/skills/hpc-alloc`, Codex at `~/.codex/skills/hpc-alloc` (or `$CODEX_HOME/skills/hpc-alloc`)

Both harnesses read the same `skill/` package. A harness counts as found when its launcher is on `PATH` or its home directory exists; `--claude` or `--codex` installs for one regardless. Finding none is a failure rather than a success that installed no skill.

`setup` validates and writes the authoritative config, initializes the SQLite state database, finds or creates an SSH key, and adds one `Include` to `~/.ssh/config`. Upload the printed public key at <https://sshkeys.ycrc.yale.edu/>, wait for propagation, then authenticate:

```bash
hpc-alloc connect
```

`connect --push` performs the same bootstrap with one Duo push, so expect a phone prompt — and expect it before letting an unattended agent run it.

Changing the config later is deliberately constrained. Stateful commands hold a shared configuration-scope lock while `setup` holds the exclusive side, and `setup --force` will not change your NetID, remove a cluster something still references, or move that cluster's login host while any job is non-final or any operation unresolved. Replacing a config within the same scope is always allowed. To change scope, run `hpc-alloc status`, finish or cancel what remains, and reconcile the printed `hpc-alloc recover OPERATION_ID` commands first.

## Running under an agent

The bundled skill teaches Claude Code and Codex how to drive hpc-alloc safely. It loads on demand, so it costs a session nothing until the cluster comes up.

Codex also sandboxes commands to the workspace and restricts network access, while hpc-alloc needs the network on every cluster-touching command and writes durable state under `~/.config/hpc-alloc/` and `~/.ssh/`. Approve those escalations when Codex asks — but keep any standing approval to the read-only verbs: `status`, `config`, `avail`, `partitions`, `why`, and `logs`. A blanket `hpc-alloc` rule would also pre-approve `run` and `ssh`, which execute arbitrary commands on the cluster, along with `sync --delete`, `cancel`, and `down --all`. Leave those to prompt every time; that prompt is the last thing standing between an agent and a cancelled seat. The bundled skill states the same division to the agent, deliberately: it obeys the prompt, while deciding what to pre-approve is yours.

A denial that interrupts a submit is an unknown remote outcome, not proof that nothing was submitted. Run `hpc-alloc status` and reconcile any unresolved operation rather than retrying.

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
| `setup [--netid NETID] [--cluster NAME] [--host HOST] [--identity-file PATH] [--force]` | Create the config, state database, key material, and managed SSH include. Replacing an existing config requires `--force`. |
| `config [--cluster NAME] [--json]` | Validate config and show the effective resource values without contacting a cluster. |
| `connect [--cluster NAME] [--reset] [--push]` | Establish or heal the login master and health-check known allocation nodes. |
| `up [--name NAME] [--cluster NAME] [resources] [--idle-timeout MIN] [--no-wait] [--wait-timeout SEC]` | Submit a persistent sleeper allocation and wait for its node. `--no-wait` returns as soon as the submission is durable, without observing the scheduler. |
| `run [--cluster NAME] [resources] [--chdir DIR] [--detach] -- CMD...` | Submit a command. Foreground mode follows output and returns the accounting exit status or the documented final-state fallback. |
| `status [--json]` | Reconcile locally journaled jobs and classify hpc-alloc-tagged queue rows across all configured clusters. |
| `why [TARGET] [--cluster NAME] [--json]` | Explain a queued, running, uncertain, or final job selected by name, job ID, or `@operation`. |
| `logs TARGET [--cluster NAME] [-n/--lines LINES] [-f/--follow]` | Read or follow a managed job log by convenience or durable selector. |
| `cancel (JOBID\|@OPERATION) [--cluster NAME]` | Cancel a managed job only after exact remote identity verification. |
| `down NAME\|JOBID\|@OPERATION\|--all [--cluster NAME]` | Cancel one or all managed allocation jobs. The target is required: `down` is irreversible, so it never guesses which allocation you meant. |
| `ssh [--cluster NAME] [NAME\|JOBID\|@OPERATION] [-- CMD...]` | Open an allocation shell or run a command there. |
| `sync (NAME\|JOBID\|@OPERATION) SRC DST [--cluster NAME] [--pull] [--delete]` | Transfer files with rsync through the allocation alias. `--pull` reverses the direction; `--delete` is destructive and needs explicit intent. |
| `avail [--cluster NAME] [-p PARTITION] [--for] [--json]` | Summarize idle CPUs and free GPUs for one cluster, marking which partitions your account may submit to. With `--for` plus a request (`-G/-c/--mem/-t/-C`), probe where that request would start soonest. |
| `partitions [--cluster NAME] [--json]` | Show live partition limits, GRES, and feature data for one cluster, plus whether your account may submit to each (`eligible`). |
| `recover [OPERATION_ID] [--cluster NAME] [--abandon] [--yes]` | Reconcile ambiguous submit/cancel operations by exact queue or accounting identity, or explicitly abandon one local intent. |

`setup --force` never re-keys. An `identity_file` already in the config is kept, because the key it names is the one registered with the cluster; change it deliberately with `--identity-file`. If a configured key has vanished from disk, setup fails rather than silently substituting one the cluster will reject.

`avail` reports what is idle at this instant, which is not a schedulability guarantee — reserved or higher-priority nodes can still queue you. Each partition carries an eligibility marker (`ELIGIBLE` in the text output, `eligible` in `--json`) so capacity you cannot submit to is not mistaken for yours. `avail --for` goes further and asks the scheduler where a given request would start soonest: a dry run that submits nothing, ranked by an estimate the queue can invalidate a moment later. Preemptible and short pools appear but are marked, since `up` and `run` never auto-select them.

`sync` hands the remote path to rsync, which expands it through the remote login shell — that is what makes `'~/project'` work. The remote path is therefore restricted to `A-Za-z0-9_@%+=:,./~-`, and one containing a space, quote, glob, or `$(...)` is rejected rather than silently re-split by that shell.

Resource flags shared by `up` and `run` are `--cluster`, `-p/--partition`, `-t/--time`, `-c/--cpus`, `--mem`, `-G/--gpus`, `-C/--constraint`, and `--dry-run`. `up` additionally accepts `--idle-timeout`, `--no-wait`, and `--wait-timeout`. Where a flag defaults, it does so quietly: `up --name` is `dev`, `up --wait-timeout` is 1800 seconds, `logs -n` is 100 lines, and `setup --cluster` is `bouchet` — the one `--cluster` anywhere with a default, since every other command resolves it from the config instead. `--idle-timeout` guards against a GPU allocation sitting idle, so it requires `-G/--gpus` and is rejected without it.

When you request a specific GPU type with `-G TYPE:N` but omit `-p/--partition`, `up` and `run` select the partition: if the default partition already offers that type it is kept, otherwise the tool auto-selects the single dedicated partition that offers it and prints which it chose; it refuses locally when several qualify, and when only preemptible or short pools offer the type it names them with `-p` guidance rather than dispatching an unschedulable request. Dedicated means a partition not matched by the cluster's `nondedicated_partition_globs` (default `scavenge*`, `*devel`). Selection reads a cached GPU-topology map and cached access rules, so a `--dry-run` resolves offline from a warm cache and prints the partition a real submit would use. It always prints a command and warns on stderr whenever the preview is not authoritative, so an unwarned command is the only one that previews the submit. It warns when it cannot resolve from the local cache at all, when a refusal holds regardless of what is cached (no partition offers the type, or only preemptible pools do), and when the cached access rules prove your account cannot use the partition it printed. It stays silent about one case on purpose: an ambiguity that exists only because the access rules are uncached, which offline cannot be distinguished from a pick the submit would make for you. A live submit instead falls open to the configured default if the topology cannot be read.

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

Queue absence is evidence, not immediate proof that a job ended. A job missing from one poll may be draining, requeueing, or hidden behind a scheduler hiccup, so the lifecycle tracker grades what it sees — no-observed-start, active, started-but-inactive, requeueing, terminal-candidate, final, uncertain — and requires confirmation before it calls anything final. Transport and scheduler failures never masquerade as job death, and final evidence is monotonic: a later weaker or missing observation cannot erase a terminal state or exit code already proved. The full taxonomy, the Slurm state mapping, and the evidence rules are in [the same reference](skill/references/recovery-and-lifecycle.md).

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
- A `phase` on a `jobs` or `why` entry is one of `QUEUED`, `ACTIVE`, `STARTED_INACTIVE`, `REQUEUEING`, `TERMINAL_CANDIDATE`, `FINAL`, `UNCERTAIN`, or `SUBMITTING`. `UNCERTAIN` reports a failed observation, not a failed job. `SUBMITTING` reports a submission with no acknowledged Slurm job ID yet — a real job may already be running, so it never means "nothing was submitted"; reconcile the printed `hpc-alloc recover OPERATION_ID` rather than resubmitting.
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
