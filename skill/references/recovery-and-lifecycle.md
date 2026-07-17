# Recovery and lifecycle

Read this reference before deciding that work is final, changing setup scope, cancelling a job, or reconciling an interrupted or ambiguous mutation.

## Protect durable authority

Treat `~/.config/hpc-alloc/state.db` and its live WAL, SHM, and rollback-journal sidecars as one SQLite state database. Let hpc-alloc validate file type, ownership, link count, and permissions; never edit, query, delete, or copy individual files to bypass the CLI.

Let stateful commands hold the shared configuration-scope lock and let `setup` hold the exclusive side through its authoritative recheck and mutations. While any durable job is non-final or operation unresolved, allow `setup --force` to replace configuration only when the NetID and every blocker-referenced cluster's normalized host stay unchanged and no blocker-referenced cluster is removed. Permit unrelated inactive-cluster changes, and permit full scope changes only after every job is final and operation resolved.

Treat each submit or cancellation as a prepare, one-shot remote call, and acknowledgement protocol using short local transactions. Let an operation-scoped advisory lock serialize its live owner with recovery and abandonment; if that lock is active, fail fast and retry only after the owning process exits. A crashed process leaves no stale lock to clear: the operating system releases it, so a lock file that outlives its owner is inert and must not be deleted by hand.

Never replay an operation whose remote outcome may be unknown. Preserve the exact operation ID and use `hpc-alloc recover OPERATION_ID`.

## Require exact identity

Recognize these identity forms:

```text
hpcalloc-v2-<alloc|run>-<32-hex-operation-id>
hpc-alloc:v2:<owner-id>:<operation-id>:<host-label>:<kind>:<logical-name-or->
```

Require the exact operation-derived Slurm name and complete persisted comment for live observation and every cancellation guard. Treat `owner-id` as authoritative and the safe host label as display metadata; reject a numeric ID, logical name, prefix, host label, legacy tag, malformed tag, or foreign-machine tag as sufficient mutation authority.

For accounting and accounting-based recovery, request full-width identity columns. Accept an empty accounting comment only with the exact operation-derived name because Bouchet may omit `Comment`; require every nonempty comment to match byte-for-byte, and fail closed on truncation or mismatch. Never apply this omission exception to a live mutation.

## Reconcile submission uncertainty

Treat submission directory preparation as idempotent and separate from the batch mutation. After the one `sbatch` call is dispatched, accept only return code 0 plus one trusted scalar job ID as acknowledgement; treat every other reply as ambiguous without retry — except the scheduler's own pre-dispatch rejection banner (`Batch job submission failed:` for a bad account, QOS, partition, constraint, or submit limit), which proves no job was created and closes the operation cleanly — the operation resolves as `FAILED` and the job records final source `submit-failed` — not an ambiguous mutation to reconcile.

If interruption rolls back the local reservation before its transaction commits, emit no recovery guidance because no operation exists and the remote submit was never entered. If interruption lands after the reservation commits but before dispatch, expect the job to be closed locally as `submit-failed` with no operation left to recover, and conservative recovery guidance only when that close cannot be confirmed. If interruption or transport loss occurs after dispatch may have begun, preserve the normal interrupt or error result, print `do not resubmit`, and use the exact recovery command; include a trusted job ID when the remote acknowledgement arrived but the local write failed. If even the ambiguity update cannot be recorded, the `PREPARED` operation stays unresolved and remains recoverable by the printed ID.

A typed authentication or host-key failure from the `sbatch` call itself is the second thing that proves no job was created: it means the command never reached the scheduler, so the operation closes as `FAILED` with no recovery guidance. Every other lost or unreadable reply stays ambiguous.

Recover a submission by finding exactly one live queue row with both the operation-derived name and complete comment, or by verifying an exact accounting row under the omission rule above. Accounting recovery reads only the last 30 days (`sacct -S now-30days`), so an older operation finds no record and stays unresolved rather than being reconciled; that is an absent lookback, not proof the job never existed. Adopt the proved job and persist final accounting through the normal lifecycle engine when applicable. Leave zero, duplicate, malformed, or identity-conflicting results unresolved, and never submit again merely because the reply was lost.

## Reconcile cancellation uncertainty

Interpret `CANCEL_PENDING` as proof that the guarded remote cancellation was not dispatched, because `AMBIGUOUS` is committed immediately before the one-shot call. Interpret `AMBIGUOUS` as may-have-run. Expect either phase to block another cancellation of the same job until recovery or explicit abandonment resolves the intent — `CANCEL_PENDING` blocks too, despite proving nothing was dispatched, so a plain retry fails with a pending-cancellation conflict rather than dispatching. Recover it instead.

On interruption before the cancellation reservation commits, emit no recovery guidance. On interruption with durable `CANCEL_PENDING`, close the undispatched intent locally when possible; if that closure cannot be confirmed, print its exact recovery command. On interruption after phase becomes `AMBIGUOUS`, preserve 130 and print the exact recovery command. Never infer dispatch from the exit status alone: `cancel` and `down` preserve 130 in every phase, so only the durable phase says whether the guarded call went out.

Recover `CANCEL_PENDING` locally without contacting Slurm. Recover an ambiguous cancellation locally when its target already has durable `ACCOUNTING` or `CONFIRMED_QUEUE` finality, including when `status` persisted that evidence after the cancellation became ambiguous. In bulk recovery, prioritize such local work and continue a bounded local sweep even if a later remote bootstrap fails.

When local evidence is insufficient, make cancellation recovery observation-only: inspect the exact live queue identity and exact final accounting, and never replay `scancel`. Resolve the operation on exact final accounting or two consecutive successful non-live observations for the old operation, requiring the two observations even with accounting for a requeue-eligible terminal (`NODE_FAIL` or `PREEMPTED`), since a single accounting record can be the reaped attempt of a job Slurm is requeueing under the same ID; accept exact absence, an exact scheduler-terminal row, or proof that the numeric ID now belongs to a different exact identity as non-live evidence.

Let an exact live row resolve the operation too, in the opposite direction. A job still observed in a state a landed cancellation could not have produced proves that cancellation never arrived, so recovery closes the operation as failed, releases the one-pending-cancel guard, and prints the `cancel` command to dispatch a fresh guarded one; repeating a cancellation is idempotent, unlike replaying a submission. Leave the operation unresolved only where the observation cannot tell the difference: a job draining through the kill sequence (`SIGNALING`, `COMPLETING`, or `STAGE_OUT`), which is exactly what a landed cancellation looks like; a terminal candidate, which is provisional death and often the cancellation itself landing; and any failed or uncertain observation.

Use `recover OPERATION_ID --cluster NAME` only when the explicit cluster equals the operation's recorded cluster; require that check before prompting, state changes, projection updates, or remote access. Report an already-resolved operation's durable phase successfully, but reject `--abandon` for resolved operations.

Treat `recover OPERATION_ID --abandon` as discarding only local intent while a remote orphan may remain. Inspect remote evidence independently, obtain explicit user approval, and use `--yes` only to skip the prompt—not to imply safety. Never abandon in bulk.

## Interpret lifecycle evidence

Use these assessment phases as policy authority:

- Treat `QUEUED` as present but not proven started.
- Treat `ACTIVE` as started and currently assigned to a node.
- Treat `STARTED_INACTIVE` as previously started but not currently active; keep it log-eligible.
- Treat `REQUEUEING` as previously started and potentially able to run again.
- Treat `TERMINAL_CANDIDATE` as one successful non-live observation, not finality; preserve any active allocation's SSH projection until finality is confirmed.
- Treat `FINAL` as durable terminal authority with accounting, confirmed-queue, submit-failed, or abandoned provenance (`final_source` serializes those in exactly that lowercase-hyphen spelling).
- Treat `UNCERTAIN` as process-local observation failure that must not overwrite prior successful durable evidence or authorize cleanup.

`status --json` and `why --json` may also report the durable phase `SUBMITTING`, which is not an assessment phase: it means the submission has no acknowledged Slurm job ID yet, so there is nothing to assess. A real job may nevertheless be running on the cluster. Never treat it as "nothing was submitted" and never resubmit; reconcile the printed `hpc-alloc recover OPERATION_ID` first, exactly as for any unresolved operation.

Map recognized Slurm queue states consistently:

- Classify `RUNNING`, `RESIZING`, and `SIGNALING` as `ACTIVE`.
- Classify `PENDING`, `CONFIGURING`, and `RESV_DEL_HOLD` as `QUEUED` before first start, or as `REQUEUEING` when durable start history already exists.
- Classify `SUSPENDED`, `STOPPED`, `COMPLETING`, and `STAGE_OUT` as `STARTED_INACTIVE`.
- Classify `REQUEUED`, `REQUEUE_FED`, `REQUEUE_HOLD`, and `SPECIAL_EXIT` as `REQUEUEING`.
- Treat recognized final scheduler rows as non-live evidence requiring confirmation. Exact final accounting resolves an ordinary terminal row immediately, but a requeue-eligible row (`NODE_FAIL` or `PREEMPTED`, which Slurm may restart under the same job ID) still needs a second independent observation, because a single accounting record can be the reaped attempt of a job being requeued.
- Treat any present but unrecognized state as `UNCERTAIN`; never let it drive log, cancellation, draining, or projection policy.

Require consecutive successful non-live evidence within one observation session. Break consecutiveness on scheduler, transport, protocol, or process-restart uncertainty. Let a present nonterminal exact row clear a prior terminal candidate, and let a lifecycle revision race rebase policy on the fresh durable row before log access, drain, sleep, or SSH projection changes.

Persist successful queue observations during long follows before diagnostics, log reads, output, drain, or sleep, and persist exact accounting enrichment before reporting it. Skip persistence when normalized lifecycle fields are unchanged, never persist `UNCERTAIN`, and keep final evidence monotonic; allow exact accounting to enrich a queue-confirmed final record without letting weaker evidence erase terminal state or exit code.

Keep `COMPLETING` and every other started-inactive state log-eligible. Permit a durably queue-confirmed scheduler-final job to make its operation-scoped log best-effort eligible even without a prior observed start. Retire allocation aliases and SSH masters only after durable finality, not on a terminal candidate, transient absence, or requeue.

Treat `why` start estimates, priority output, and reservation listings as optional enrichment after an identity-checked core assessment. Omit ordinary scheduler or remote-command enrichment failures, but propagate authentication and host-key failures. Persist applicable delayed accounting before rendering; keep elapsed time and time limit as output-only data and drop them if reconciliation makes the accounting record stale.

Treat `SUBMIT_FAILED` and `ABANDONED` as local final verdicts that may lack a Slurm job ID. Let `why @operation` diagnose them without cluster access, and let `logs @operation` fail locally because no managed remote log is confirmed. Reserve recovery guidance for a genuinely unresolved submission.

## Expect patience, not immediacy

The polling commands ride out transient failures instead of failing on them. A scheduler hiccup is retried for up to two minutes and a dropped transport for up to ten, so `up`, `run`, and `logs -f` survive a controller restart, a VPN renegotiation, or a closed laptop lid rather than aborting while the job keeps running. A command that takes longer than expected is usually waiting on purpose; read its stderr, which says what it is retrying and for how long. Only an authentication or host-key failure is raised immediately, because time cannot heal it.

Observations of a steady job also widen over time — up to thirty seconds between scheduler queries — while log streaming keeps its own faster cadence. A job that changes state is noticed promptly, because any change collapses the interval back to its floor. Do not infer from a few seconds of silence that a command has hung or that a job has stalled.

## Interpret status identity classes

Read `status` managed jobs and discovered queue evidence separately. For one exact operation match, classify a durable final local record as `local-final-conflict` before considering whether it lacks a job ID; reserve `unresolved-operation-match` for a genuinely unresolved no-ID submission. Preserve `duplicate-operation` and `operation-identity-conflict` when cardinality or identity disagrees, and never convert any discovered classification into cancellation authority.

On an unavailable secondary cluster, preserve its local rows as `UNCERTAIN` and continue reporting other clusters. Treat primary-cluster failure as fatal, and treat a host-key change on any cluster as a hard integrity failure rather than degraded availability.
