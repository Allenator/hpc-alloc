# hpc-alloc

`hpc-alloc` submits ordinary Slurm batch jobs on Yale YCRC clusters and makes
running allocation jobs reachable from a laptop over SSH. It is intended for
interactive development on partitions where `salloc` is unavailable, as well
as short-lived batch commands whose output should stream back to the caller.

The v2 implementation is a deliberate clean cut. It requires Python 3.11 or
newer and does not read, import, or manage v1 configuration, `state.json`, job
names, or Slurm comments.

## How it works

For `hpc-alloc up`, the client submits a sleeper with `sbatch`, waits for Slurm
to assign a node, and writes an SSH alias such as `hpc-bouchet.dev`. The alias
ProxyJumps through the cluster login host. Login and compute-node connections
use OpenSSH multiplexing, so one Duo authentication can serve later polling,
SSH, and rsync commands.

For `hpc-alloc run`, the client submits the requested command as a normal batch
job. A foreground run follows the output and returns the Slurm job's exit
status; `--detach` leaves it running for a later `hpc-alloc logs ... -f`.

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

`install.sh` verifies Python before linking the executable into
`~/.local/bin` and the bundled Claude Code skill into
`~/.claude/skills/hpc-alloc`. These are symlinks, so keep the checked-out
repository at its installed path.

`setup` validates and writes the authoritative v2 config, initializes the
SQLite state database, finds or creates an SSH key, and adds one `Include` to
`~/.ssh/config`. Upload the printed public key at
<https://sshkeys.ycrc.yale.edu/>, wait for propagation, then authenticate:

```bash
hpc-alloc connect
```

`connect --push` performs the same bootstrap with one Duo push. Tell the user
to expect and approve that push before invoking it from an unattended agent.

### Clean cut from v1

There is no migration path in the executable. A v1 `state.json` is ignored,
and jobs carrying a v1 name or comment are not considered owned. Replace an
old config with `hpc-alloc setup --force --netid YOUR_NETID`; archive or remove
old files separately if they are no longer needed. Existing v1 jobs must be
handled outside v2 rather than adopted by a heuristic match.

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

Use `up --dry-run` or `run --dry-run` to print the `sbatch` command without
connecting or changing local state.

## Commands

| Command | Purpose |
|---|---|
| `setup --netid NETID [--cluster NAME] [--host HOST]` | Create the v2 config, state database, key material, and managed SSH include. Existing config requires `--force`. |
| `config [--cluster NAME] [--json]` | Validate config and show the effective resource values without contacting a cluster. |
| `connect [--cluster NAME] [--reset] [--push]` | Establish or heal the login master and health-check known allocation nodes. |
| `up [--name NAME] [resources]` | Submit a persistent sleeper allocation and wait for its node. `--no-wait` leaves it queued. |
| `run [resources] [--chdir DIR] [--detach] -- CMD...` | Submit a command. Foreground mode follows output and mirrors the job result. |
| `status [--json]` | Reconcile locally journaled jobs and classify v2-tagged queue rows across all configured clusters. |
| `why [TARGET] [--cluster NAME] [--json]` | Explain a queued, running, uncertain, or final job selected by name, job ID, or `@operation`. |
| `logs TARGET [-n LINES] [-f]` | Read or follow a managed job log by convenience or durable selector. |
| `cancel JOBID\|@OPERATION` | Cancel a managed job only after exact remote identity verification. |
| `down [NAME\|@OPERATION\|--all]` | Cancel one or all managed allocation jobs. |
| `ssh [NAME\|@OPERATION] [-- CMD...]` | Open an allocation shell or run a command there. |
| `sync NAME\|@OPERATION SRC DST [--pull] [--delete]` | Transfer files with rsync through the allocation alias. |
| `avail [-p PARTITION] [--json]` | Summarize idle CPUs and free GPUs for one cluster. |
| `partitions [--json]` | Show live partition limits, GRES, and feature data for one cluster. |
| `recover [OPERATION_ID] [--cluster NAME]` | Reconcile ambiguous submit/cancel operations by exact queue or accounting identity. |

Resource flags shared by `up` and `run` are `--cluster`, `-p/--partition`,
`-t/--time`, `-c/--cpus`, `--mem`, `-G/--gpus`, `-C/--constraint`, and
`--dry-run`. `up` additionally accepts `--idle-timeout`, `--no-wait`, and
`--wait-timeout`.

## Authoritative configuration

The only application config is `~/.config/hpc-alloc/config.toml`. Its schema is
strict: unknown tables, unknown keys, invalid types, and invalid values are
errors. See [`config.example.toml`](config.example.toml) for a complete example.

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

`[identity].netid` and at least one `[cluster.NAME].host` are required. The
`[ssh]` and `[defaults]` tables may be empty. Every resource key accepted in
`[defaults]` may also appear in a cluster table; a cluster value overrides the
global default. Invocation precedence is:

```text
CLI flag > selected [cluster.NAME] value > [defaults] value > built-in fallback
```

If several clusters are configured, commands that must choose a cluster
implicitly require `[defaults].cluster` (notably `status`) or `--cluster` where
that flag is supported. A cluster-qualified job selector supplies its own
cluster, while unfiltered `recover` and `down --all` may span clusters without
a default. There is no fallback host convention in the config parser: each
cluster table must declare its host explicitly.

## Durable job selectors

The operation ID is the durable identity; Slurm job IDs can be recycled and
logical names can repeat. The canonical selector is:

```text
@operation_id
cluster:@operation_id
```

Commands also accept the convenience forms:

```text
cluster:name
cluster:jobid
```

Examples are `bouchet:@08a3a68f1ad04ac595836695e0e9cc95`, `bouchet:dev`, and
`bouchet:123456`. Numeric and name selectors prefer one current non-final job
over retained history. Other ambiguity errors list the operation selectors
that remain individually addressable. A qualifier and `--cluster` must agree.
Numeric IDs are temporary locators: when Slurm reuses one, the old operation is
reconciled only through its exact operation-derived accounting identity and the
replacement is reported separately. Jobs are never rebound by numeric ID.

`status` polls every configured cluster. Failure of the primary cluster is an
error. An unavailable secondary is reported on stderr, its local records are
preserved as `UNCERTAIN`, and no absence-based cleanup is performed. Commands
that act on one job use that job's cluster. `down --all` can span clusters;
`--cluster` restricts it. A host-key change is an integrity failure on every
cluster and always aborts status rather than degrading to `UNCERTAIN`.

## Durable state and recovery

Tool-owned state lives in `~/.config/hpc-alloc/state.db`, a mode-0600 SQLite
database using WAL mode. SQLite may create `state.db-wal` and `state.db-shm`
sidecars while the database is open. Do not edit or copy individual files from
an active database. This release uses a clean-cut state schema and provides no
migration; archive or remove an older database only after accounting for any
still-running remote jobs, then run setup again.

The database records the machine identity, jobs, lifecycle evidence and its
final source, cluster caches, and a durable operation journal. Submission and
cancellation follow a prepare/remote-call/acknowledge protocol with short local
transactions:

1. Persist the intended mutation and its exact identity.
2. Leave the SQLite transaction before invoking SSH or Slurm.
3. Perform the remote mutation once.
4. Record the acknowledgement, or mark the operation ambiguous if the reply
   may have been lost.

An ambiguous mutation is never guessed safe to retry. `status` exposes it and
prints the relevant operation ID. Reconcile it with:

```bash
hpc-alloc recover OPERATION_ID
```

Recovery requires the exact operation-derived v2 job name. A live queue match
must also contain the complete expected comment. Accounting reads explicitly
request full-width identity columns; Bouchet accounting may still omit
`Comment`, so an empty accounting comment is accepted only with the exact job
name. Any nonempty accounting comment must match the persisted comment
byte-for-byte, so truncated or mismatched identities fail closed. If there is
still no conclusive match, the operation remains unresolved.

Submission directory preparation is a separate idempotent step. Once the one
batch submission call is dispatched, only a successful reply containing one
trusted scalar job ID is an acknowledgement; every other reply remains
ambiguous even when the remote command returned nonzero.

Recovery does not replay an ambiguous cancellation. It performs read-only
queue and accounting observations and never issues another `scancel`. A live
exact match or inconclusive absence remains visible for operator follow-up;
only exact final accounting or two successful exact queue absences closes the
pending cancellation.

The cancellation journal records dispatch certainty explicitly:
`CANCEL_PENDING` means the guarded remote call was never dispatched, while
`AMBIGUOUS` is committed immediately before that one call and means it may have
run. Both phases block another cancellation for the same job. To authorize a
new attempt after an ambiguous result, first inspect the remote state and then
explicitly abandon the old operation.

`recover OPERATION_ID --abandon` discards only the local intent and warns that
a remote orphan may remain; it prompts for confirmation unless `--yes` is
present.

`SUBMIT_FAILED` and `ABANDONED` are durable local final verdicts and may have no
Slurm job ID. `why @operation` reports either verdict entirely from local state.
`logs @operation` exits 1 locally because there is no confirmed managed remote
log; it does not contact the cluster or recommend `recover`. A genuinely
unresolved submission remains `SUBMITTING`, and both commands instead print the
exact recovery operation.

### Exact ownership

Each mutation gets a random 32-hex-character operation ID. A v2 Slurm job uses
both of these identifiers:

```text
job name: hpcalloc-v2-<alloc|run>-<operation_id>
comment:  hpc-alloc:v2:<owner_id>:<operation_id>:<host_label>:<kind>:<logical_name-or->
```

Live observations and mutation guards, including cancellation, require both
the exact job name and complete comment. Terminal accounting and accounting
recovery use the narrower omission rule above because `sacct` may return an
empty comment. A matching numeric job ID, prefix, logical name, or machine
host label alone is never sufficient. Host labels are safe deterministic
display metadata; `owner_id` is authoritative. Jobs with malformed, legacy, or
foreign live tags are not cancelled.

## Lifecycle and stream policy

Queue absence is evidence, not immediate proof that a job ended. The lifecycle
tracker distinguishes never-started, active, started-but-inactive, requeueing,
terminal-candidate, final, and uncertain states. Candidate termination is
confirmed with another successful observation, and final accounting is used
when available; transport and scheduler failures do not masquerade as job
death. A present `COMPLETING` job is started-but-inactive, remains log-eligible,
and cannot become final merely because it appears in two consecutive polls.
Recycled numeric IDs count as non-live evidence only for the old exact
operation; partial name/comment matches remain hard identity conflicts.

Final evidence is monotonic. Later missing or weaker observations cannot erase
persisted terminal state or exit code; exact accounting may enrich a
queue-confirmed final record.

Foreground stream behavior is intentionally command-specific:

- Ctrl-C during `run` cancels that exact job and the CLI returns 130.
- A closed stdout pipe during foreground `run` also initiates cancellation and
  returns 141.
- Ctrl-C during `logs -f` detaches; the job continues and the CLI returns 130.
- A closed stdout pipe during `logs -f` detaches, leaves the job running, and
  returns 141.
- A clean `logs -f` returns 0 regardless of the job's final Slurm state.
- A foreground `run` returns the batch command's nonzero exit code; any
  non-`COMPLETED` final state is nonzero.

Progress and recovery notices go to stderr. JSON stdout remains machine-only.
Transport/authentication failures that require human intervention use exit 3;
configuration, scheduler, protocol, and other application failures use exit 1.

## JSON contracts

The v2 JSON surfaces are intentionally explicit:

- `config --json` returns `config_file`, `state_file`, `primary_cluster`, the
  validated `config`, and `effective` resource values.
- `status --json` returns exactly three top-level arrays: `jobs`, `discovered`,
  and `operations`. `jobs` carry a canonical `selector` plus operation,
  scheduler, lifecycle, terminal, resource, node, and alias fields. A job
  finalized during reconciliation appears once in `jobs`, not again as a
  discovered conflict. `discovered` separates `job_kind` (`allocation` or
  `run`) from `classification` (`untracked-owned`, `other-machine`,
  `unresolved-operation-match`, `duplicate-operation`,
  `local-final-conflict`, or `operation-identity-conflict`). `operations`
  contains unresolved submit/cancel journal rows, their target-job `selector`,
  and recovery details.
- `why --json` returns one job assessment and diagnosis.
- `avail --json` returns `{ "partitions": { ... } }`.
- `partitions --json` returns an array of partition objects.

Consumers should use `kind` for managed jobs, `job_kind` plus `classification`
for discovered rows, and canonical `selector` values for later actions. Do not
parse display text. There are no v1 JSON aliases.

## GPU policy

Prefer `hpc-alloc run -G TYPE:N -- ...` for GPU work so the GPU is held only
while the command executes. A persistent GPU allocation created with `up -G`
installs an idle watchdog; the built-in threshold is 30 consecutive minutes at
at most 5% utilization. Configure `idle_timeout`, override it with
`--idle-timeout MIN`, or explicitly pass `0` to disable the watchdog.

`hpc-alloc avail` reports current free capacity, and `partitions` supplies the
live partition/GRES names. Request only the resources needed: larger requests
usually wait longer and affect fair share.

## Files managed

- `~/.config/hpc-alloc/config.toml` — authoritative user configuration
- `~/.config/hpc-alloc/state.db*` — SQLite state, WAL, and transient sidecar
- `~/.config/hpc-alloc/ssh_config` — lock-serialized regenerated login/allocation aliases
- `~/.config/hpc-alloc/known_hosts` — managed host-key store, namespaced by cluster and physical node
- `~/.ssh/config` — only an `Include` for the managed SSH config is added
- `~/.ssh/hpc-alloc-*` — OpenSSH control sockets
- `~/.hpc-alloc/` on each cluster — operation-ID-named allocation and run logs

## Troubleshooting

- Exit 3 means the VPN/SSH authentication path needs attention. Reconnect the
  VPN, run `hpc-alloc connect`, and retry. A scheduler error over a healthy SSH
  connection is exit 1; reconnecting cannot repair Slurm.
- If `status` lists an unresolved operation, run the exact `recover` command it
  prints. Do not submit or cancel again just because an SSH reply was lost.
- If a name or job ID is ambiguous, use the reported `cluster:@operation_id` selector.
- A host-key change is surfaced as a hard failure. Verify the new key with YCRC
  before changing known-hosts data.
- Run `hpc-alloc why TARGET` for pending reasons or final accounting evidence.
- Walltime cannot be extended. Synchronize important work before the deadline.

## Tests

```bash
./tests/run.sh
```

The suite is offline and uses strict Python test doubles and framed-protocol
fixtures. It covers configuration validation, SQLite permissions and
transactions, concurrent mutation invariants, exact ownership, ambiguous
submit/cancel recovery, SSH/Slurm failure classification, lifecycle traces,
requeue-aware streaming, multi-cluster selectors, status identity-graph
behavior, and tracked-only package delivery. Release validation must use a
Git-index or committed clean-tree export; green tests in a dirty workspace do
not prove that newly created package files are included. The suite does not
require a cluster, VPN, credentials, or network access.
