---
name: hpc-alloc
description: Allocate and use Yale YCRC compute nodes for development, submit CPU/GPU batch commands, synchronize files, diagnose jobs, reconcile ambiguous mutations, and safely release exact v2-owned jobs. Use when a user asks to inspect, allocate, use, recover, or release YCRC cluster resources.
---

# hpc-alloc v2

Use `hpc-alloc` to manage YCRC Slurm work from the user's laptop. `up`
submits a sleeper batch job and exposes its compute node through a managed SSH
alias; `run` submits a finite command and optionally streams its log.

This skill describes v2 only. It requires Python 3.11+ and has no compatibility
with v1 config, `state.json`, job names, ownership comments, or JSON aliases.
Never infer ownership from a name, numeric job ID, prefix, or hostname.

## Connection and failure protocol

An active Yale VPN connection is required. The user must establish the VPN.
One-time setup is:

```bash
hpc-alloc setup --netid NETID
```

`setup --force` is scope-safe. While any durable job is non-final or operation
is unresolved, it may replace configuration only with the same NetID and the
same normalized host for every blocker-referenced cluster; it may not remove
those clusters. Run `hpc-alloc status`, finish or cancel remaining jobs, and
use each printed `hpc-alloc recover OPERATION_ID` command before changing
scope. Unrelated inactive clusters may change, and all scope changes are
allowed after every job is final and every operation resolved.

The user uploads the printed key at <https://sshkeys.ycrc.yale.edu/> and then
runs `hpc-alloc connect` in a terminal.

Exit 3 means the SSH/VPN path requires human attention. Do not blind-retry.
Ask the user to run `hpc-alloc connect`, or first tell them to expect a Duo
push and invoke:

```bash
hpc-alloc connect --push
```

Send at most one expected push. A Slurm/scheduler or protocol error over a
healthy connection exits 1; `connect` cannot fix it. Report that distinction.
Jobs remain on the cluster during client-side transport failures.

## Start with structured status

Before allocating or acting on an existing job, run:

```bash
hpc-alloc status --json
```

Stdout is JSON; progress and unavailable-secondary notes are on stderr. The
top-level object has exactly these arrays:

- `jobs`: locally managed jobs reconciled during this pass. A job finalized
  during the pass appears here once with `phase: "FINAL"`; it is not repeated
  as discovered. Important fields include canonical `selector`,
  `operation_id`, `jobid`, `cluster`, `name`, `kind`, lifecycle and terminal
  fields, resources, and `alias`.
- `discovered`: validated v2 queue rows not consumed as the normal bound row
  for a live managed job.
  `job_kind` is `allocation` or `run`; `classification` is one of
  `untracked-owned`, `other-machine`, `unresolved-operation-match`,
  `duplicate-operation`, `local-final-conflict`, or
  `operation-identity-conflict`. Each row also has its canonical `selector`.
  Treat every discovered row as evidence, not permission to cancel.
- `operations`: unresolved submit/cancel journal records. Important fields are
  `operation_id`, target-job `selector`, `kind`, `phase`, `cluster`, `target`,
  `jobid`, and `detail`.

Do not expect v1 fields such as `allocs`, heuristic `orphan`, or `recent`.

Lifecycle phases distinguish queued, active, previously started but inactive,
requeueing, final candidates, final jobs, and uncertain evidence. A queue
absence alone does not mean a job died. If a secondary cluster is unavailable,
its local jobs remain `UNCERTAIN`; never reap them based on that absence.
Final evidence is monotonic: missing or weaker observations cannot erase a
persisted terminal state or exit code, while exact accounting may enrich a
queue-confirmed final record. A `why` retry that discovers delayed accounting
persists its final source, state, and exit code before reporting them;
`Elapsed` and `Timelimit` are output-only enrichment and are discarded if the
accounting result is no longer applicable after reconciliation.

A present `COMPLETING` row is started-but-inactive and log-eligible, not final
evidence. Numeric job IDs may be recycled: reconcile the old job through its
exact operation identity, show the replacement separately, and never rebind or
mutate by numeric ID alone. `@operation` remains the canonical historical
selector.

## Commands

| Command | Use |
|---|---|
| `hpc-alloc config --json` | Validate authoritative config and inspect effective defaults without cluster access. |
| `hpc-alloc status --json` | Reconcile all configured clusters and inspect local jobs, classified v2-tagged rows, and unresolved operations. Run first. |
| `hpc-alloc avail [--cluster C] [-p PART] [--json]` | Inspect idle CPUs and free GPUs before requesting scarce resources. |
| `hpc-alloc partitions [--cluster C] [--json]` | Inspect live partitions, limits, GRES, and features. |
| `hpc-alloc up [--name N] [resources]` | Create a persistent development allocation. It waits for a node unless `--no-wait` is given. |
| `hpc-alloc run [resources] [--chdir DIR] [--detach] -- CMD...` | Run a finite batch command. Foreground mode follows output and mirrors the job result. Prefer this for GPU execution. |
| `hpc-alloc logs TARGET [-n N] [-f]` | Read or follow a managed job log. TARGET may be a name, job ID, or `@operation`. Following never implicitly cancels. |
| `hpc-alloc why [TARGET] [--json]` | Diagnose pending, active, uncertain, or final lifecycle evidence by convenience or durable selector. |
| `hpc-alloc ssh [NAME\|@OPERATION] -- CMD...` | Run a command on an active allocation. Omitting CMD opens an interactive user shell. |
| `hpc-alloc sync NAME\|@OPERATION SRC DST [--pull] [--delete]` | Transfer files through the allocation alias with rsync. |
| `hpc-alloc cancel JOBID\|@OPERATION` | Cancel an exact managed allocation or run after live identity verification. |
| `hpc-alloc down [NAME\|@OPERATION\|--all]` | Cancel exact managed allocation jobs. |
| `hpc-alloc recover [OPERATION_ID]` | Reconcile ambiguous remote mutation replies using exact queue/accounting identity. |
| `hpc-alloc connect [--reset] [--push]` | Establish or heal SSH masters. |

Shared resource flags are `--cluster`, `-p/--partition`, `-t/--time`,
`-c/--cpus`, `--mem`, `-G/--gpus`, and `-C/--constraint`. Pass only values the
task requires; let authoritative config supply the rest. Use `--dry-run` to
inspect an `up` or `run` submission without connecting or changing state.

Numeric `--time` values accept `minutes`, `minutes:seconds`,
`hours:minutes:seconds`, `days-hours`, `days-hours:minutes`, or
`days-hours:minutes:seconds`. Minute and second subfields must be two digits
from `00` through `59`. Do not use signs, whitespace, `INFINITE`, or
`UNLIMITED`. All-zero spellings are also invalid because Slurm treats zero as
a request for no time limit; always choose a finite nonzero duration.

## Durable job selectors

The operation ID is the durable identity. Slurm can recycle numeric job IDs,
and logical names can repeat. Prefer the canonical form for reattachment,
history, and automation:

```text
@operation_id
cluster:@operation_id
```

The CLI also accepts convenience forms:

```text
cluster:name
cluster:jobid
```

Examples: `bouchet:@08a3a68f1ad04ac595836695e0e9cc95`, `bouchet:dev`, and
`bouchet:123456`. Name and numeric selectors prefer one current non-final job
over retained history. If ambiguity remains, the error lists canonical
operation selectors. A qualifier and `--cluster` must agree.

`status` polls all configured clusters and needs a configured primary when
several exist. Other read commands act on the cluster supplied by their parsed
selector or explicit flag; a qualified selector does not require a default.
Unfiltered recovery and `down --all` may span clusters, and `--cluster`
restricts them.

## Durable mutation recovery

State is a SQLite WAL database at `~/.config/hpc-alloc/state.db`. Do not edit
it, query it to bypass the CLI, delete its `-wal`/`-shm` files, or copy only one
live sidecar. The journal persists a mutation before the remote call and
records the acknowledgement afterward. If SSH fails after a submit or cancel
may have committed, v2 marks the operation ambiguous and does not guess that a
retry is safe.

When `status` reports an unresolved operation, run the printed command:

```bash
hpc-alloc recover OPERATION_ID
```

If Ctrl-C or SIGTERM interrupts submission after dispatch may have begun,
preserve exit 130 and follow the printed `do not resubmit` recovery guidance.
When Slurm returned a trusted job ID but the local acknowledgement write
failed, the notice includes that ID. Even if the ambiguity marker itself
cannot be written, the durable `PREPARED` operation remains recoverable; never
replace this workflow with a fresh submission.

For `recover OPERATION_ID --cluster NAME`, the explicit cluster must equal the
cluster recorded on the operation. A mismatch fails before confirmation,
abandonment, local projection changes, or remote access. Explicit recovery of
an already-resolved operation reports its durable phase successfully, while
`--abandon` still rejects it.

Recovery requires the exact operation-derived v2 Slurm name. A live queue row
must also match the complete persisted comment. Accounting reads request
full-width identity columns. Bouchet accounting may omit `Comment`; only an
empty accounting comment may be accepted with the exact name. Any nonempty
accounting comment must match byte-for-byte, and any truncated name or
nonempty-comment mismatch fails closed. If recovery cannot prove the outcome,
leave the operation unresolved and report it; do not repeat the original
mutation.

Cancellation recovery is observation-only. It checks the exact live queue row
and final accounting, never repeats `scancel`, and leaves the cancellation
pending when those reads cannot prove the outcome. A later explicit cancel is
a separate user-authorized mutation, not part of `recover`. `CANCEL_PENDING`
means no cancellation call was dispatched; `AMBIGUOUS` is committed before the
one guarded call and blocks another attempt until the operation is explicitly
abandoned.

Before cancelling an already-absent job, require two successful exact queue
absences or exact final accounting. Submission preparation contains no batch
mutation; after the one batch call is dispatched, every reply except rc0 plus
one trusted scalar job ID is ambiguous and must be recovered without retry.

`hpc-alloc recover OPERATION_ID --abandon` discards only the local journal
intent and can leave a remote orphan. Use it only after explicit user approval
and after the remote outcome has been independently checked. `--yes` skips its
confirmation prompt but does not make abandonment safer.

`SUBMIT_FAILED` and `ABANDONED` are local final verdicts, not unresolved
operations, even though they may have no Slurm job ID. `why @operation` reports
them without cluster access. `logs @operation` exits 1 locally because no
managed remote log is confirmed and must not suggest `recover`. Only a genuinely
unresolved `SUBMITTING` record gets recovery guidance.

V2 identities have these forms:

```text
hpcalloc-v2-<alloc|run>-<32-hex-operation-id>
hpc-alloc:v2:<owner-id>:<operation-id>:<host-label>:<kind>:<logical-name-or->
```

Live observations and cancellation guards require the exact managed job name
and complete comment before `scancel`. The empty-comment exception applies
only to terminal accounting and accounting recovery, never to a live mutation.
The safe host label is display metadata; `owner-id` is the authoritative
machine identity.
Never call `scancel` directly and never attempt to adopt or cancel legacy,
malformed, foreign-machine, or merely similar live jobs.

## Stream and signal policy

- Foreground `run` returns the batch command's exit code. Any final Slurm state
  other than `COMPLETED` is nonzero.
- Ctrl-C during foreground `run` cancels its exact job; the CLI returns 130.
- A closed stdout pipe during foreground `run` initiates cancellation and
  returns 141. If cancellation itself is ambiguous, report the recovery ID.
- Ctrl-C or a closed pipe during `logs -f` only detaches. The job continues;
  Ctrl-C returns 130 and BrokenPipe returns 141.
- A clean `logs -f` returns 0 regardless of the job's final state.

Do not interpret a pipe closure as ordinary success. Do not replace `logs -f`
with foreground `run` when the desired policy is detach-on-client-exit.

## Resource discipline

Read effective values when needed:

```bash
hpc-alloc config --json
```

Precedence is CLI flag, selected `[cluster.NAME]`, `[defaults]`, then built-in
fallback. The strict v2 config requires `[identity].netid` and an explicit
`host` in every `[cluster.NAME]` table.

For GPU work:

1. Run `hpc-alloc avail --json` and, if necessary, `partitions --json` to get
   current capacity and exact GRES names.
2. Prefer `hpc-alloc run -G TYPE:N -- ...`; it holds GPUs only while the
   command runs.
3. Use a CPU `up` allocation for editing/building and submit GPU execution with
   `run` when practical.
4. A persistent `up -G ...` allocation has an idle watchdog. Do not disable it
   with `--idle-timeout 0` without the user's explicit approval, and never
   create fake GPU load to evade cluster policy.

Set `--mem` explicitly for memory-heavy work. Shorter realistic walltimes can
improve backfill opportunities, and larger requests usually queue longer and
consume more fair share. Use `why` rather than guessing why a job is pending.

## Typical workflow

1. `hpc-alloc status --json`; recover unresolved operations and reuse a
   suitable active allocation.
2. `hpc-alloc config --json` and `avail --json` if resource selection matters.
3. Create a development seat with `hpc-alloc up --name dev`, adding only the
   required resource flags.
4. Push code with `hpc-alloc sync bouchet:dev ./project '~/project'`.
5. Build/test with
   `hpc-alloc ssh bouchet:dev -- 'cd ~/project && make test'`. Load modules in
   the remote command because compute nodes start with a minimal environment.
6. Submit GPU work with `run`; use `--detach` for long work and follow by its
   reported `cluster:@operation` selector.
7. Pull results with `sync --pull`.
8. Ask before `down` or cancellation unless releasing work was already part of
   the user's instruction.

## Safety rules

- Never run heavy computation on the login node.
- Never directly edit `~/.config/hpc-alloc/state.db` or its WAL sidecars.
- Never directly call `scancel`; use `cancel` or `down` for exact verification
  and durable journaling.
- Never retry an ambiguous submit/cancel. Use `recover`, which observes and
  reconciles evidence without replaying an ambiguous cancellation.
- Never claim a job ended from a transport failure, one absent queue sample, or
  an unavailable secondary cluster.
- Never cancel a `discovered` entry or an `other-machine` job merely
  because it has a v2-looking tag. Report it and obtain explicit direction.
- Treat host-key changes as a hard stop until the user verifies the new key
  through a trusted YCRC channel. This applies equally to primary and secondary
  clusters; compute keys and control masters are namespaced by cluster.
- Walltime is a hard deadline and cannot be extended. Synchronize important
  outputs before it expires.
