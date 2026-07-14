# Command contracts

Use this reference to choose exact invocations and interpret public CLI results. Keep stdout machine-only when requesting JSON; send progress, recovery instructions, and degraded-secondary notices to stderr.

## Command surface

| Command | Contract |
|---|---|
| `hpc-alloc setup --netid NETID [--cluster C] [--host HOST] [--identity-file PATH] [--force]` | Validate and write the authoritative v2 configuration, initialize state, find or create key material, and install the managed SSH include. Require `--force` to replace an existing config. `--force` repairs a NetID or host mistake and never re-keys: a configured `identity_file` is preserved, since that key is the one registered with the cluster. Re-key only with an explicit `--identity-file`. |
| `hpc-alloc config [--cluster C] [--json]` | Validate configuration and report effective resource values without cluster access. |
| `hpc-alloc connect [--cluster C] [--reset] [--push]` | Establish or heal login masters and health-check known allocation nodes; use `--push` only after warning the user about one Duo push. |
| `hpc-alloc up [--name N] [resources] [--idle-timeout MIN] [--no-wait] [--wait-timeout SEC]` | Submit a persistent sleeper allocation. Wait for an active node unless `--no-wait` returns immediately after durable submission acknowledgement. Exit 0 means a node is ready; exit 4 means the wait expired with the job still queued — it is submitted and tracked, so wait rather than resubmit. `--idle-timeout` guards an idle GPU allocation and requires `-G/--gpus`; it is rejected without it. |
| `hpc-alloc run [resources] [--chdir DIR] [--detach] -- CMD...` | Submit a finite batch command. Follow its operation-scoped log in foreground mode or return after acknowledged submission with `--detach`. |
| `hpc-alloc status [--json]` | Reconcile locally journaled jobs and classify v2-tagged queue rows across every configured cluster. |
| `hpc-alloc why [TARGET] [--cluster C] [--json]` | Diagnose the selected queued, active, inactive, uncertain, or final job and persist any applicable delayed accounting evidence. |
| `hpc-alloc logs TARGET [--cluster C] [-n LINES] [-f]` | Read or follow an operation-scoped managed log. Wait safely for start with `-f`; never make follow imply cancellation. |
| `hpc-alloc cancel JOBID\|@OPERATION [--cluster C]` | Cancel a managed allocation or run only after exact live identity verification; reject logical-name selectors. |
| `hpc-alloc down NAME\|JOBID\|@OPERATION\|--all [--cluster C]` | Cancel one or all managed allocation jobs with exact verification and durable journaling. The target is required and is never inferred: a bare `down` exits 1 and lists the active allocations. |
| `hpc-alloc ssh [--cluster C] [NAME\|JOBID\|@OPERATION] [-- CMD...]` | Open an interactive shell or replace the client process with SSH running a command on one active allocation. Place `--cluster` before the target because the remaining arguments belong to SSH. |
| `hpc-alloc sync NAME\|JOBID\|@OPERATION SRC DST [--cluster C] [--pull] [--delete]` | Run rsync through one active allocation alias; treat `--delete` as destructive and require explicit intent. |
| `hpc-alloc avail [--cluster C] [-p PARTITION] [--json]` | Summarize currently idle CPUs and free GPUs for one cluster. |
| `hpc-alloc partitions [--cluster C] [--json]` | Report live partition limits, GRES, and feature data for one cluster. |
| `hpc-alloc recover [OPERATION_ID] [--cluster C] [--abandon] [--yes]` | Reconcile unresolved mutations without replaying them; restrict abandonment to one explicit operation and obtain confirmation unless `--yes` is explicitly authorized. |

Apply the shared `up` and `run` resource flags `--cluster`, `-p/--partition`, `-t/--time`, `-c/--cpus`, `--mem`, `-G/--gpus`, `-C/--constraint`, and `--dry-run`. Resolve resource values in this order: CLI flag, selected `[cluster.NAME]`, `[defaults]`, then built-in fallback.

Accept numeric durations as `minutes`, `minutes:seconds`, `hours:minutes:seconds`, `days-hours`, `days-hours:minutes`, or `days-hours:minutes:seconds`. Require two-digit minute and second subfields from `00` through `59`; reject signs, whitespace, symbolic unlimited values, and every all-zero spelling.

Use `--dry-run` to print a paste-ready `up` or `run` submission command without connecting or changing state. Execute the printed command in the target login shell so `${HOME:?}`, relative paths, and `~/...` paths resolve for that remote account.

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
- Read `why --json` as one job assessment and diagnosis, `avail --json` as `{ "partitions": { ... } }`, and `partitions --json` as an array of partition objects.
- Reject assumptions about v1 fields such as `allocs`, heuristic `orphan`, `recent`, or legacy JSON aliases.

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
