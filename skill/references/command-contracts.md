# Command contracts

Use this reference to choose exact invocations and interpret public CLI results. Keep stdout machine-only when requesting JSON; send progress, recovery instructions, and degraded-secondary notices to stderr.

## Command surface

| Command | Contract |
|---|---|
| `hpc-alloc setup [--netid NETID] [--cluster C] [--host HOST] [--identity-file PATH] [--force]` | Validate and write the authoritative configuration, initialize state, find or create key material, and install the managed SSH include. Always pass `--netid`: argparse does not require it, so an omitted NetID prompts on a terminal and fails as a clean exit 1 anywhere else, including under an agent. Require `--force` to replace an existing config. `--force` repairs a NetID or host mistake and never re-keys: a configured `identity_file` is preserved, since that key is the one registered with the cluster. Re-key only with an explicit `--identity-file`. |
| `hpc-alloc config [--cluster C] [--json]` | Validate configuration and report effective resource values without cluster access. |
| `hpc-alloc connect [--cluster C] [--reset] [--push]` | Establish or heal login masters and health-check known allocation nodes; use `--push` only after warning the user about one Duo push. |
| `hpc-alloc up [--name N] [resources] [--idle-timeout MIN] [--no-wait] [--wait-timeout SEC]` | Submit a persistent sleeper allocation. Wait for an active node unless `--no-wait` returns immediately after durable submission acknowledgement. Exit 0 means a node is ready; exit 4 means the wait expired with the job still queued — it is submitted and tracked, so wait rather than resubmit. `--idle-timeout` guards an idle GPU allocation and requires `-G/--gpus`; it is rejected without it. |
| `hpc-alloc run [resources] [--chdir DIR] [--detach] -- CMD...` | Submit a finite batch command. Follow its operation-scoped log in foreground mode or return after acknowledged submission with `--detach`. |
| `hpc-alloc status [--json]` | Reconcile locally journaled jobs and classify hpc-alloc-tagged queue rows across every configured cluster. |
| `hpc-alloc why [NAME\|JOBID\|@OPERATION] [--cluster C] [--json]` | Diagnose the selected queued, active, inactive, uncertain, or final job and persist any applicable delayed accounting evidence. |
| `hpc-alloc logs NAME\|JOBID\|@OPERATION [--cluster C] [-n LINES] [-f]` | Read or follow an operation-scoped managed log; `-n` defaults to 100 lines. Wait safely for start with `-f`; never make follow imply cancellation. |
| `hpc-alloc cancel JOBID\|@OPERATION [--cluster C]` | Cancel a managed allocation or run only after exact live identity verification; reject logical-name selectors. |
| `hpc-alloc down NAME\|JOBID\|@OPERATION\|--all [--cluster C]` | Cancel one or all managed allocation jobs with exact verification and durable journaling. The target is required and is never inferred: a bare `down` exits 1 and lists the active allocations. |
| `hpc-alloc ssh [--cluster C] [NAME\|JOBID\|@OPERATION] [-- CMD...]` | Open an interactive shell or replace the client process with SSH running a command on one active allocation. Place `--cluster` before the target because the remaining arguments belong to SSH. |
| `hpc-alloc sync NAME\|JOBID\|@OPERATION SRC DST [--cluster C] [--pull] [--delete]` | Run rsync through one active allocation alias; treat `--delete` as destructive and require explicit intent. |
| `hpc-alloc avail [--cluster C] [-p PARTITION] [--json]` | Summarize currently idle CPUs and free GPUs for one cluster. Idle GRES is not a schedulability guarantee — reserved or higher-priority-tier nodes can still queue you. Each partition also carries an access-eligibility marker (`ELIGIBLE` column; `eligible` in `--json`) that is true, false, or null when the access data is unavailable, so idle capacity you cannot submit to is not mistaken for yours. |
| `hpc-alloc avail --for [-G G] [-c N] [--mem M] [-t T] [-C X] [-p P] [--json]` | Probe the scheduler (a dry-run that submits no job) for where the given request would start soonest, across the eligible, GRES-matching partitions, ranked by estimated start. Estimates are advisory (the queue shifts). Restrict to one partition with `-p` — an ineligible `-p` is reported as not-schedulable, never refused, because this is a read-only probe. Otherwise the eligible candidates are probed, ordered by free capacity and bounded to a fixed count (the text says so when it caps the set; `--json` carries a `capped` boolean). Preemptible or short pools are still shown but marked, since `up`/`run` will not auto-select them. A `-G TYPE:N` whose type no partition offers is reported as an unknown-or-unavailable GPU type (listing the known types) so a typo reads differently from nothing being free. |
| `hpc-alloc partitions [--cluster C] [--json]` | Report live partition limits, GRES, and feature data for one cluster; each partition also carries an `eligible` flag (true/false, or null when access data is unavailable) for whether the account, QOS, and groups may submit to it. |
| `hpc-alloc recover [OPERATION_ID] [--cluster C] [--abandon] [--yes]` | Reconcile unresolved mutations without replaying them; restrict abandonment to one explicit operation and obtain confirmation unless `--yes` is explicitly authorized. |

Apply the shared `up` and `run` resource flags `--cluster`, `-p/--partition`, `-t/--time`, `-c/--cpus`, `--mem`, `-G/--gpus`, `-C/--constraint`, and `--dry-run`. Resolve resource values in this order: CLI flag, selected `[cluster.NAME]`, `[defaults]`, then built-in fallback. When `-G TYPE:N` is given without `-p`, `up` and `run` steer the resolved default partition to one that actually offers that GPU type: the default is kept if it already offers the type, otherwise the single dedicated partition offering it is auto-selected and announced, the command refuses locally when several qualify, and when only preemptible or short pools offer it, it names them with `-p` guidance rather than claiming nothing offers the type. Dedicated means a partition not matched by the cluster's `nondedicated_partition_globs` config (fnmatch globs; default `scavenge*` and `*devel`). Resolution reads a cached GPU-topology map, so `--dry-run` resolves offline from a warm cache — printing the same partition a real submit would. It always prints a command, and warns on stderr whenever it could not resolve one: on a cold cache it says so and prints the configured default, and on a warm cache that proves the pick ambiguous or impossible it prints the same refusal a real submit would raise, then the configured default. Read the warning rather than assuming a cold cache; the printed partition is only authoritative when no warning appears. Before dispatching, `up` and `run` refuse locally only a partition the account, QOS, or groups PROVABLY cannot use: a fail-open accelerator that reads cached access rules (warmed by `connect`), so a clear access error is caught before the round-trips without a fetch. It falls open on any uncertainty (missing, empty, or partition-scoped data), and the scheduler's own verdict is the authoritative gate — a deterministic rejection (bad account, QOS, partition, constraint, or submit limit) is returned as a clean local failure (exit 1), not an ambiguous mutation to recover.

Accept numeric durations as `minutes`, `minutes:seconds`, `hours:minutes:seconds`, `days-hours`, `days-hours:minutes`, or `days-hours:minutes:seconds`. Require exactly two digits from `00` through `59` in every subfield that follows a colon, and leave a field no colon precedes unbounded and unpadded (`5`, `90:30`, and `100:00:00` all validate). Reject signs, whitespace, symbolic unlimited values, and every all-zero spelling.

Use `--dry-run` to print a paste-ready `up` or `run` submission command without connecting or changing state. Execute the printed command in the target login shell so `${HOME:?}`, relative paths, and `~/...` paths resolve for that remote account. A command you paste and run yourself creates a real job that hpc-alloc does not journal or track; its comment carries a `dryrun-` tag precisely so `status` and recovery never mistake it for a tool-managed job.

## Selectors and cluster scope

Treat the operation ID as durable identity and use either canonical form:

```text
@operation_id
cluster:@operation_id
```

Use convenience selectors only when appropriate:

```text
name
jobid
cluster:name
cluster:jobid
```

Allow name selectors only where the command contract permits them. Prefer one current non-final job over retained history for name and numeric selectors; when ambiguity remains, use one of the canonical selectors printed by the error. Require a selector qualifier and explicit `--cluster` to agree.

Never treat a numeric job ID as durable ownership. When Slurm recycles it, reconcile the old job through its exact operation identity, show the replacement separately, and never rebind or mutate the old record by the number alone.

Poll every configured cluster for `status`; require a configured primary when several clusters make implicit selection ambiguous. Let other single-job reads use a selector qualifier or `--cluster`; let unfiltered `recover` and `down --all` span clusters, and use `--cluster` to restrict them.

Allow an exact durable-final selector to resolve for local history after `setup --force` removes its recorded cluster. Treat that as local selection only: any path that still needs SSH, scheduler, accounting, or log access requires the recorded cluster to remain configured and never falls back to another cluster.

## JSON contracts

Treat JSON stdout as the stable machine surface and do not parse display text.

- Read `config --json` as an object containing `config_file`, `state_file`, `primary_cluster`, validated `config`, and `effective` resource values.
- Read `status --json` as exactly three top-level arrays: `jobs`, `discovered`, and `operations`.
- Read each `jobs` entry through its canonical `selector`, `operation_id`, `jobid`, `cluster`, `name`, `kind`, lifecycle and terminal fields, resources, node, and alias. Keep a job finalized during the pass in `jobs` once rather than repeating it in `discovered`.
- Read each `discovered` entry through `job_kind` plus `classification`; accept `untracked-owned`, `other-machine`, `unresolved-operation-match`, `duplicate-operation`, `local-final-conflict`, and `operation-identity-conflict` as classification values, and treat its canonical selector as evidence rather than mutation authority.
- Read each `operations` entry through `operation_id`, target-job `selector`, `kind`, `phase`, `cluster`, `target`, `jobid`, and `detail`.
- Read `why --json` as one job assessment and diagnosis, `avail --json` as `{ "partitions": { ... } }` where each partition object carries an `eligible` flag (true, false, or null when access data is unavailable), `avail --for --json` as `{ "for": { resolved request }, "probes": [ { "partition", "preemptible", "schedulable", "start", "detail" } ], "capped": bool }` ordered soonest-first, where `capped` is true when more eligible partitions existed than were probed (with an added `error` string, `capped` false, and empty `probes` when the requested GPU type is unknown or unavailable), and `partitions --json` as an array of partition objects, each carrying an `eligible` flag (true, false, or null when access data is unavailable).
- Rely only on the documented JSON fields; do not assume any other keys exist.

## Exit, stream, and signal policy

Treat argparse usage failures, including missing required arguments and invalid typed values, as exit 2; they print usage and occur before command dispatch. Interpret a post-parse hpc-alloc validation, scheduler, protocol, or application failure as exit 1. Interpret typed authentication, host-key, or transport failures as exit 3, but inspect stderr and command context because exit statuses can be passed through.

Interpret exit 4 from `up` as "submitted, not ready yet": the wait expired while the allocation was still queued. This is neither success nor failure. The job is submitted, durable, and tracked, so never resubmit it — poll `hpc-alloc status`, follow it with `hpc-alloc logs CLUSTER:JOBID -f`, or release it with `hpc-alloc down NAME`. On a busy GPU partition this is an ordinary outcome, not an error.

Let foreground `run` return the numeric batch exit status when exact accounting provides it. When confirmed queue finality has no accounting exit status, return 0 for `COMPLETED` and 1 otherwise; coerce any non-`COMPLETED` final state with numeric status 0 to 1.

Allow `run` to pass through any numeric application exit, including 2 or 3. Allow `ssh` to replace the client with OpenSSH and `sync` to return rsync's status, including 2 or 3; do not diagnose those statuses as parser or VPN failures without supporting stderr and command context.

- On Ctrl-C or SIGTERM while `up` waits after submission, preserve 130, do not cancel the allocation, and use the printed canonical selector with `status` or `down`.
- On an ordinary scheduler, transport, or log failure while foreground `run` follows, do not cancel the submitted job; use the printed selector to reattach with `logs -f`, cancel it, or inspect an already durable-final record.
- On Ctrl-C or SIGTERM while foreground `run` follows, attempt exact cancellation and preserve 130 whether cancellation succeeds, fails, or becomes ambiguous; follow any printed recovery command.
- On a closed stdout pipe while foreground `run` follows, attempt exact cancellation and return 141; follow any printed recovery command when cancellation cannot be confirmed.
- On Ctrl-C, SIGTERM, or a closed pipe during `logs -f`, detach without cancellation; return 130 for the interrupt or 141 for the broken pipe.
- On a clean `logs -f` completion, return 0 regardless of the job's final Slurm state.
- On interruption during direct `cancel` or `down`, close an undispatched cancellation intent when durable phase proves dispatch never began; after dispatch may have begun, preserve 130 and follow the exact printed recovery command.

Do not interpret a pipe closure as ordinary success, and do not substitute foreground `run` for `logs -f` when detach-on-client-exit is required.
