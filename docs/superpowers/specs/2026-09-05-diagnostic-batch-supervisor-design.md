# Diagnostic Batch Supervisor Design

## Goal

Run diagnostic session batches sequentially in isolated experiment processes, retain every batch outcome, and produce a merged diagnostic-only result only when all batches have complete profile coverage.

## Execution Model

The supervisor materializes complete-session fixtures as it does today. It runs one `run_experiment.py pilot-smoke-measured` child process per batch, serially. A child exit is recorded rather than immediately terminating the supervisor, so a trace-semantics gate failure in one diagnostic batch cannot prevent coverage collection from later batches.

Each child keeps its own experiment output directory. This preserves process-level GPU cleanup between batches and prevents session state from accumulating across the 27-session diagnostic fixture.

## Batch Status

The supervisor writes a status record for every planned batch containing its identity, input paths, child exit code, expected request count, expected profile names, detected profile-row count, coverage result, and the child output directory.

A batch is mergeable only when its child exits successfully or exits solely because diagnostic gates did not pass, its profile CSV exists, every expected `(request_id, profile)` pair occurs exactly once, and every row has `ok=true` and `measured=true`. Other nonzero exits, missing artifacts, malformed CSV, incomplete coverage, duplicate keys, or failed rows make the batch non-mergeable.

## Merge Contract

After all child processes finish, the supervisor writes a top-level summary manifest regardless of success. It writes merged artifacts only when every planned batch is mergeable:

- `merged/profile_tables/diagnostic_session27_profiles.csv`, containing all validated profile rows.
- `merged/session_traces/diagnostic_session27_trace.csv`, when every batch trace is present.
- `merged/manifest.json`, with `diagnostic_only: true`, source batch paths, validation counts, and merge status.

The merged data retains every original row and appends no formal-baseline provenance. It is never a canonical fixture, formal smoke table, or policy-comparison input. A non-mergeable run exits nonzero after writing its status manifest and does not emit partial merged CSVs.

## Error Handling

The supervisor distinguishes expected diagnostic gate exits from execution failures by inspecting the batch summary and gate artifacts. A trace/risk gate failure with complete profile coverage is recorded as `diagnostic_gate_failed` and remains mergeable. Any other failed child status is an execution failure.

## Tests

Tests will use temporary fixtures and fake child runners. They will prove that a diagnostic gate exit does not prevent later batches, complete batches merge in deterministic batch order, and an incomplete batch prevents merged output while leaving the top-level status manifest.
