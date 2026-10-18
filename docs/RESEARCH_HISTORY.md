# Research history and evidence preservation

The formal project integrates the original team context, an initial individual benchmark, and the later performance investigation into one repository entry point. Internal package names (`distress_ml`, `distress_v2`) and historical candidate IDs remain stable for import and serialized-model compatibility.

## Preserved commits

- Previous default-branch tip: `b9e78c6454bfc7cbac6560bdfb047589dc8cd2fe`.
- Reviewed research tip: `75baf24e2b0388cd6b201ddbb9b6006683e6b6e7`.
- Scientific publication commit: `e7f6fb51238730bb5e055572b752e2ff0c84b102`.

The research tip is merged with a two-parent merge commit before the layout consolidation. No history is rewritten. A local all-refs Git bundle and untracked-source backup were made before consolidation. Original local data, private audit packages, models, clocks, and source workspaces remain outside this public checkout.

## What is kept

All later aggregate CSV/JSON results, per-trial configurations, negative/incomplete attempts, registered protocol bytes, selection freeze, and artifact inventory remain unchanged. [Aggregate hash manifest](research_origin/aggregate_manifest.json) binds the relocated files to their original contents. The original English and Chinese PDFs are byte-identical copies with clearer public filenames. Their internal research-stage titles and original report sources in [frozen reporting](research_origin/frozen_reporting/) remain historical records.

Earlier aggregate evidence and configurations are retained under [initial extension](research_origin/initial_extension/). Its old reports and notebooks remain retrievable from the original Git commits; they are not parallel current deliverables. The [search record](research_origin/search_record/), [protocol execution notes](research_origin/search_record/PROTOCOL_EXECUTION_NOTES.md), and [full baseline bridge](../results/baseline_bridge.csv) document source roles, budget limits, migration corrections, and negative outcomes. Historical operational scripts are retained only as source records, not as supported user commands.

## Research boundaries

The first eight-hour run expired with a partial result. A separately owner-authorized continuation reused its completed evidence and preserved both clocks. All 1,312 recorded model-fit starts were verified inside those windows. This consolidation performs public evidence checks, software tests, figure rendering from saved aggregates, and notebook execution; it does not refit or reselect the frozen models.

The earlier preservation receipt established equality for 1,344 readable protected files; one private original was denied by macOS permissions. It did not claim full equality for that unreadable file. The present release retains that limitation and makes no fresh claim to have audited inaccessible originals.

Two archive-role prediction permutations were corrected without fitting or selection changes. Their original diagnoses and correction receipts remain public aggregates. The invalidated oversampling migration attempt, incomplete trial coverage, and unsuccessful optimization controls are not removed or relabeled as successes.

## Layout compatibility

The statistical protocol now lives at `configs/protocol.yaml` with its original hash. Model and result paths in frozen configuration remain relative to a run root. The current package resolves the repository root or an explicit `DISTRESS_PROJECT_ROOT`; the shared initial package resolves its historical context under `docs/research_origin/initial_extension`. Optional private source reconstruction uses explicit `DISTRESS_SOURCE_ROOT` and `DISTRESS_INITIAL_ROOT` paths. Current user commands are documented in [reproduction](REPRODUCTION.md).
