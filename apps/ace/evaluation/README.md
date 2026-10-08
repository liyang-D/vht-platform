# ACE local automated evaluation

`apps/ace/ace-eval` is the non-Web, operator-only entry point. Every ordinary
run creates a timestamped directory under `ACE_ROOT/evaluation/batches/`; use
`--batch` to add selected evidence to an imported batch or resume it.

```bash
apps/ace/ace-eval list
apps/ace/ace-eval import-existing
apps/ace/ace-eval run --batch BATCH --group unit
apps/ace/ace-eval run --batch BATCH --test classify_turn
apps/ace/ace-eval run --case 'unit.exact::test'
apps/ace/ace-eval run --group vision --range 23:44
apps/ace/ace-eval run --all
apps/ace/ace-eval report --batch BATCH
```

Ranges are one-based and inclusive, applied after any group/test selection.
Completed pass and failed assertions are skipped on resume; interrupted,
execution-error and missing cases receive a new attempt. Exit codes are `0`
success, `1` test failures, `2` selection/usage error, `3` infrastructure or
integrity failure, and `130` interrupt. `--rerun-complete` must be explicit to
repeat completed evidence; reports retain every attempt and never select the
best result.

The worker runs inside the local ACE backend image on `vht-dev_default` and
uses only the local orchestrator LLM/VLM endpoints. It does not expose an HTTP
API and does not invoke cloud or alternative vision models.
