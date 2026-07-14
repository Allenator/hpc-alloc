# hpc-alloc

`hpc-alloc` submits ordinary Slurm batch jobs on Yale YCRC clusters and makes running allocation jobs reachable from a laptop over SSH. It is intended for interactive development on partitions where `salloc` is unavailable, as well as short-lived batch commands whose output should stream back to the caller.

The v2 implementation is a deliberate clean cut. It requires Python 3.11 or newer and does not read, import, or manage v1 configuration, `state.json`, job names, or Slurm comments.

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

`install.sh` verifies Python, the launcher, the complete bundled skill package, and the complete runtime-module manifest under an isolated Python startup before creating any links. It then links the executable into `~/.local/bin` and the bundled Claude Code skill into `~/.claude/skills/hpc-alloc`. These are symlinks, so keep the checked-out repository at its installed path.

`setup` validates and writes the authoritative v2 config, initializes the SQLite state database, finds or creates an SSH key, and adds one `Include` to `~/.ssh/config`. Upload the printed public key at <https://sshkeys.ycrc.yale.edu/>, wait for propagation, then authenticate:

```bash
hpc-alloc connect
```

Stateful commands hold a shared configuration-scope lock, while `setup` holds the exclusive side through its authoritative recheck and all mutations. A forced setup cannot change the NetID, remove a blocker-referenced cluster, or change that cluster's normalized login host while any job is non-final or any operation is unresolved. Same-scope replacement remains allowed. Use `hpc-alloc status`, finish or cancel remaining jobs, and run the printed `hpc-alloc recover OPERATION_ID` commands before changing scope. Once every job is final and every operation resolved, `setup --force` may change it.

`connect --push` performs the same bootstrap with one Duo push. Tell the user to expect and approve that push before invoking it from an unattended agent.

### Clean cut from v1

There is no migration path in the executable. A v1 `state.json` is ignored, and jobs carrying a v1 name or comment are not considered owned. Replace an old config with `hpc-alloc setup --force --netid YOUR_NETID` only after the v2 journal has no setup blockers; archive or remove old files separately if they are no longer needed. Existing v1 jobs must be handled outside v2 rather than adopted by a heuristic match.

## Common workflow

```bash
# Inspect local journal entries and exact v2 jobs found on every cluster.
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

Use `up --dry-run` or `run --dry-run` to print a paste-ready submission command without connecting or changing local state. The command keeps the remote home as `${HOME:?}`, so execute it in the target login shell; relative paths and `~/...` working directories then resolve beneath that account's home directory.

## Commands

| Command | Purpose |
|---|---|
| `setup [--netid NETID] [--cluster NAME] [--host HOST] [--force]` | Create the v2 config, state database, key material, and managed SSH include. Existing config requires `--force`. |
| `config [--cluster NAME] [--json]` | Validate config and show the effective resource values without contacting a cluster. |
| `connect [--cluster NAME] [--reset] [--push]` | Establish or heal the login master and health-check known allocation nodes. |
| `up [--name NAME] [--cluster NAME] [resources] [--idle-timeout MIN] [--no-wait] [--wait-timeout SEC]` | Submit a persistent sleeper allocation. The default waits for a node; `--no-wait` returns after durable submission acknowledgement without observing the scheduler state. |
| `run [--cluster NAME] [resources] [--chdir DIR] [--detach] -- CMD...` | Submit a command. Foreground mode follows output and returns the accounting exit status or the documented final-state fallback. |
| `status [--json]` | Reconcile locally journaled jobs and classify v2-tagged queue rows across all configured clusters. |
| `why [TARGET] [--cluster NAME] [--json]` | Explain a queued, running, uncertain, or final job selected by name, job ID, or `@operation`. |
| `logs TARGET [--cluster NAME] [-n LINES] [-f]` | Read or follow a managed job log by convenience or durable selector. |
| `cancel (JOBID\|@OPERATION) [--cluster NAME]` | Cancel a managed job only after exact remote identity verification. |
| `down [NAME\|JOBID\|@OPERATION\|--all] [--cluster NAME]` | Cancel one or all managed allocation jobs. |
| `ssh [--cluster NAME] [NAME\|JOBID\|@OPERATION] [-- CMD...]` | Open an allocation shell or run a command there. |
| `sync (NAME\|JOBID\|@OPERATION) SRC DST [--cluster NAME] [--pull] [--delete]` | Transfer files with rsync through the allocation alias. |
| `avail [--cluster NAME] [-p PARTITION] [--json]` | Summarize idle CPUs and free GPUs for one cluster. |
| `partitions [--cluster NAME] [--json]` | Show live partition limits, GRES, and feature data for one cluster. |
| `recover [OPERATION_ID] [--cluster NAME] [--abandon] [--yes]` | Reconcile ambiguous submit/cancel operations by exact queue or accounting identity, or explicitly abandon one local intent. |

Resource flags shared by `up` and `run` are `--cluster`, `-p/--partition`, `-t/--time`, `-c/--cpus`, `--mem`, `-G/--gpus`, `-C/--constraint`, and `--dry-run`. `up` additionally accepts `--idle-timeout`, `--no-wait`, and `--wait-timeout`.

Numeric Slurm durations support all six documented forms: `minutes`, `minutes:seconds`, `hours:minutes:seconds`, `days-hours`, `days-hours:minutes`, and `days-hours:minutes:seconds`. Minute and second subfields must be two digits from `00` through `59`; signs, whitespace, and symbolic values such as `INFINITE` or `UNLIMITED` are not accepted. Every all-zero spelling is also rejected because Slurm interprets a zero duration as requesting no time limit; specify an explicit, finite nonzero duration instead.

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
mem = "16G"
idle_timeout = 30

[cluster.bouchet]
host = "bouchet.ycrc.yale.edu"
```

`[identity].netid` and at least one `[cluster.NAME].host` are required. The `[ssh]` and `[defaults]` tables may be empty. Every resource key accepted in `[defaults]` may also appear in a cluster table; a cluster value overrides the global default.

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

Commands that accept a managed job target also accept the convenience forms:

```text
cluster:name
cluster:jobid
```

Examples are `bouchet:@08a3a68f1ad04ac595836695e0e9cc95`, `bouchet:dev`, and `bouchet:123456`. `why`, `logs`, `down`, `ssh`, and `sync` accept names, numeric job IDs, and operation selectors; `cancel` accepts only a numeric job ID or operation selector. Numeric and name selectors prefer one current non-final job over retained history. Other ambiguity errors list the operation selectors that remain individually addressable. A qualifier and `--cluster` must agree. Numeric IDs are temporary locators: when Slurm reuses one, the old operation is reconciled only through its exact operation-derived accounting identity and the replacement is reported separately. Jobs are never rebound by numeric ID.

An exact durable selector for a retained final record remains meaningful after a permitted setup removes its cluster. Locally answerable history such as a no-ID `SUBMIT_FAILED` or `ABANDONED` diagnosis remains accessible by the recorded `cluster:@operation` selector. Any action or history read that needs SSH, scheduler, accounting, or remote-log access still requires that recorded cluster to be present in the current configuration and never falls back to another cluster.

`status` polls every configured cluster. Failure of the primary cluster is an error. An unavailable secondary is reported on stderr, its local records are preserved as `UNCERTAIN`, and no absence-based cleanup is performed. Commands that act on one job use that job's cluster. `down --all` can span clusters; `--cluster` restricts it. A host-key change is an integrity failure on every cluster and always aborts status rather than degrading to `UNCERTAIN`.

## Durable state and recovery

Tool-owned state lives in `~/.config/hpc-alloc/state.db`, a mode-0600 SQLite database using WAL mode. SQLite may create `state.db-wal`, `state.db-shm`, and a rollback journal. Do not edit or copy individual files from an active database. This release uses a clean-cut state schema and provides no migration; archive or remove an older database only after accounting for any still-running remote jobs, then run setup again.

The database records the machine identity, jobs, lifecycle evidence and its final source, cluster caches, and a durable operation journal. Submission and cancellation follow a prepare/remote-call/acknowledge protocol with short local transactions:

1. Persist the intended mutation and its exact identity.
2. Leave the SQLite transaction before invoking SSH or Slurm.
3. Perform the remote mutation once.
4. Record the acknowledgement, or mark the operation ambiguous if the reply may have been lost.

The live submit or cancel process owns a secure operation-scoped advisory lock from intent publication through remote dispatch and the final journal update. Recovery and abandonment acquire that same lock without waiting, then reload the durable operation. If its live owner is still dispatching or journaling, they fail closed and ask you to retry after that process exits. The operating system releases the lock if a process crashes.

An ambiguous mutation is never guessed safe to retry. `status` exposes it and prints the relevant operation ID. Reconcile it with:

```bash
hpc-alloc recover OPERATION_ID
```

Submission directory preparation is a separate idempotent step. Once the one batch submission call is dispatched, only a successful reply containing one trusted scalar job ID is an acknowledgement; every other reply remains ambiguous even when the remote command returned nonzero.

If submission is interrupted before its reservation commits, the transaction rolls back, no operation exists, and the scheduler call was not entered. If an acknowledged reservation is interrupted before remote dispatch, hpc-alloc tries to close it as `SUBMIT_FAILED`; if that local close cannot be confirmed, it prints conservative recovery guidance. If dispatch may have begun, hpc-alloc prints the exact recovery command and a `do not resubmit` warning before preserving exit 130. The guidance includes the trusted Slurm job ID when the remote acknowledgement arrived but its local journal write failed. If even the ambiguity update cannot be recorded, the existing `PREPARED` operation remains unresolved and recoverable by the printed ID.

With an explicit operation ID, `recover --cluster NAME` validates the requested cluster against the operation's recorded cluster before prompting, changing local state, or contacting a cluster. A mismatch fails closed. Recovering an already-resolved operation reports its durable phase and succeeds; `--abandon` continues to reject resolved operations.

Submission recovery requires the exact operation-derived v2 job name. A live queue match must also contain the complete expected comment. Accounting reads explicitly request full-width identity columns; Bouchet accounting may still omit `Comment`, so an empty accounting comment is accepted only with the exact job name. Any nonempty accounting comment must match the persisted comment byte-for-byte, so truncated or mismatched identities fail closed. If there is still no conclusive match, the operation remains unresolved.

The cancellation journal records dispatch certainty explicitly. `CANCEL_PENDING` means the guarded remote call was never dispatched, while `AMBIGUOUS` is committed immediately before the one guarded call and means it may have run. Both phases block another cancellation for the same job until they are resolved or explicitly abandoned.

An interrupt before a cancellation intent commits leaves no operation. An interrupt while the durable cancellation is still `CANCEL_PENDING` closes that undispatched intent locally when possible. An interrupt after the journal reaches `AMBIGUOUS` preserves the intent and prints the exact `recover` command. Direct `cancel` and `down` preserve the normal interrupt status 130 in every phase; never infer dispatch from the exit status alone.

Recovery never replays an ambiguous cancellation and never issues another `scancel`. A `CANCEL_PENDING` intent is closed locally because its guarded mutation was not dispatched. An `AMBIGUOUS` cancellation whose target is already durably final from exact accounting or confirmed queue evidence is also resolved locally before any service bootstrap. Otherwise recovery performs read-only queue and accounting observations. Exact final accounting or two consecutive successful exact non-live observations close the cancellation; non-live evidence may be queue absence, an exact scheduler-terminal row, or proof that the numeric ID now belongs to another identity. A live exact match, one non-live observation, or any failed or inconclusive observation leaves the operation unresolved.

For bulk recovery, locally resolvable cancellations are processed before operations that need a cluster, and remaining local candidates receive a best-effort sweep if a later remote recovery fails. This prevents an offline or expired-authentication cluster from stranding cancellation intents already settled by durable evidence.

`recover OPERATION_ID --abandon` discards only the local intent and warns that a remote orphan may remain; it prompts for confirmation unless `--yes` is present.

`SUBMIT_FAILED` and `ABANDONED` are durable local final verdicts and may have no Slurm job ID. `why @operation` reports either verdict entirely from local state. `logs @operation` exits 1 locally because there is no confirmed managed remote log; it does not contact the cluster or recommend `recover`. A genuinely unresolved submission remains `SUBMITTING`, and both commands instead print the exact recovery operation.

### Exact ownership

Each mutation gets a random 32-hex-character operation ID. A v2 Slurm job uses both of these identifiers:

```text
job name: hpcalloc-v2-<alloc|run>-<operation_id>
comment:  hpc-alloc:v2:<owner_id>:<operation_id>:<host_label>:<kind>:<logical_name-or->
```

Live observations and mutation guards, including cancellation, require both the exact job name and complete comment. Terminal accounting and accounting recovery use the narrower omission rule above because `sacct` may return an empty comment. A matching numeric job ID, prefix, logical name, or machine host label alone is never sufficient. Host labels are safe deterministic display metadata; `owner_id` is authoritative. Jobs with malformed, legacy, or foreign live tags are not cancelled.

## Lifecycle and stream policy

Queue absence is evidence, not immediate proof that a job ended. The lifecycle tracker distinguishes no-observed-start, active, started-but-inactive, requeueing, terminal-candidate, final, and uncertain states. Candidate termination is confirmed with another successful observation, and final accounting is used when available; transport and scheduler failures do not masquerade as job death. A present `COMPLETING` or `STAGE_OUT` job is started-but-inactive, remains log-eligible, and cannot become final merely because it appears in two consecutive polls. `RESIZING` and `SIGNALING` are active states. `RESV_DEL_HOLD` is queued before first start and requeueing after start. `SPECIAL_EXIT` is requeueing. Recycled numeric IDs count as non-live evidence only for the old exact operation; partial name/comment matches remain hard identity conflicts. An exact scheduler-final job remains eligible for a best-effort read of its operation-scoped log even when no earlier observation proved that it started.

Final evidence is monotonic. Later missing or weaker observations cannot erase persisted terminal state or exit code; exact accounting may enrich a queue-confirmed final record. When `why` discovers delayed exact accounting, it persists the accounting final source, terminal state, and exit code before rendering its diagnosis. `Elapsed` and `Timelimit` remain display enrichment and are omitted if reconciliation finds that the accounting result no longer applies.

For a valid identity-checked `PENDING` observation, `why` preserves the core diagnosis if an optional start estimate, priority lookup, or reservation listing fails because of an ordinary scheduler or transport error. It omits only the failed enrichment. Authentication requirements and host-key changes remain fatal instead of being hidden as missing detail.

Foreground and follow behavior is intentionally command-specific:

- Ctrl-C or SIGTERM while `up` is waiting does not cancel the acknowledged allocation. The CLI prints its canonical selector plus `status` and `down` guidance, then returns 130; the allocation may remain queued or running.
- Ctrl-C or SIGTERM during foreground `run` attempts to cancel the exact job, reports whether cancellation was confirmed, and returns 130. A cancellation that may have dispatched prints its operation ID and exact recovery command.
- A closed stdout pipe during foreground `run` likewise attempts exact cancellation and returns 141; inability to confirm cancellation is reported without replacing that status.
- An ordinary scheduler, transport, or log-read failure during foreground `run` does not cancel the job. The CLI prints the canonical selector and reattach/cancel guidance if it may continue, or logs/diagnosis guidance if durable finality was already reached, then propagates the original error.
- Ctrl-C or SIGTERM during `logs -f` detaches; the job continues and the CLI returns 130.
- A closed stdout pipe during `logs -f` detaches, leaves the job running, and returns 141.
- A clean `logs -f` returns 0 regardless of the job's final Slurm state.
- A foreground `run` returns the batch command's accounting exit code when available. Without an accounting exit code, it returns 0 for `COMPLETED` and 1 for another final state; it also forces a nonzero result for a non-`COMPLETED` state even if accounting reports status 0.

Progress and recovery notices go to stderr. JSON stdout remains machine-only. Argparse usage failures, including missing required arguments and invalid typed values, print usage and exit 2 before command dispatch; post-parse configuration, validation, scheduler, protocol, and other hpc-alloc application failures normally use exit 1. Exit 2 and exit 3 are contextual rather than globally reserved: a foreground batch command may itself return either status, while `ssh` and `sync` can return delegated OpenSSH or rsync statuses. Typed authentication, host-key, and transport failures use exit 3, and a possibly dispatched cancellation may surface its recovery guidance through a transport-class failure. Interpret these statuses together with the invoked command and stderr.

## JSON contracts

The v2 JSON surfaces are intentionally explicit:

- `config --json` returns `config_file`, `state_file`, `primary_cluster`, the validated `config`, and `effective` resource values.
- `status --json` returns exactly three top-level arrays: `jobs`, `discovered`, and `operations`. `jobs` carry a canonical `selector` plus operation, scheduler, lifecycle, terminal, resource, node, and alias fields. A job finalized during reconciliation appears once in `jobs`, not again as a discovered conflict. `discovered` separates `job_kind` (`allocation` or `run`) from `classification` (`untracked-owned`, `other-machine`, `unresolved-operation-match`, `duplicate-operation`, `local-final-conflict`, or `operation-identity-conflict`). `operations` contains unresolved submit/cancel journal rows, their target-job `selector`, and recovery details.
- `why --json` returns one job assessment and diagnosis.
- `avail --json` returns `{ "partitions": { ... } }`.
- `partitions --json` returns an array of partition objects.

Consumers should use `kind` for managed jobs, `job_kind` plus `classification` for discovered rows, and canonical `selector` values for later actions. Do not parse display text. There are no v1 JSON aliases.

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
