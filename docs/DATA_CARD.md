# Data card

## Scope and identity

This independent extension uses the user's existing `american_bankruptcy.csv`: SHA-256 `cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e`. Both local raw copies match. Structural audit: 78,682 firm-years, 8,971 anonymized companies, 1999-2018, 18 financial inputs, 5,220 positive rows and 609 positive companies. A firm-year is one accounting record for one company in one fiscal year, not an independent bankruptcy event.

## Label and timing

`failed=1`, `alive=0`; every company's observed label is constant across its records. The task is retrospective failed-company classification. Actual event dates, statement publication dates and when the ultimate label became knowable are unavailable. Temporal input separation cannot repair label availability. No one-year event horizon or real-time annual probability of default is established. The final observed year is not used as an event date. Financial losses remain valid negative inputs.

## Variables and units

See `research_origin/initial_extension/results/public/feature_registry.csv` and `research_origin/initial_extension/results/public/financial_field_audit.csv`. Definitions were checked against the original project dictionary. X17 means total liabilities, not exclusively interest-bearing debt. X9 and X16 are exact duplicates. Currency, monetary scaling, inflation adjustments, accounting restatements and point-in-time availability are not independently verified. Ratios use current assets/denominators; no average assets or missing cash-flow variables are invented. The Altman formula is an interpretable descriptive benchmark across a universe lacking a verified manufacturing restriction.

## Coverage and quality

There are no raw missing cells or duplicate company-years. Engineered ratios can be missing: nonpositive denominators are invalidated and flagged; extreme positive denominators/ratios are retained before train-only winsorization. 710 companies have calendar gaps. Exact t-1/t-2 joins preserve these gaps; no backward filling or future lifecycle summaries are permitted. All target records remain in model comparisons. Historical inputs for each target company are permitted only through its target year.

## Initial extension roles and exposure

Fit 1999-2012; Platt calibration 2013; fixed F1 threshold 2014; previously inspected historical evaluation 2015-2018. Development uses four expanding calendar folds. See `split_audit.csv` for independently calculated row and company counts. The historical evaluation contains 119 positive companies. Only 16 positive companies were unseen through 2012, so unseen-company estimates are especially uncertain. Three deterministic company groups provide additional untuned sensitivity checks, not external validation.

The formal project refits the selected recipes and strong baselines through 2014. Its natural unseen-company cohort contains only five failed companies; the earlier cutoff-2012 numbers above belong to the initial extension. See [current methods](METHODOLOGY.md).

## Acquisition and rights

Candidate upstream: https://github.com/sowide/bankruptcy_dataset . Its README describes 8,262 companies and a pre-event target, which conflicts with the observed company-constant labels and counts in its current CSV and the local copy. See `upstream_verification.json` for live hash and common-column comparison, rather than relying on that description. The current upstream license states CC BY 4.0. The source data, processed copies, row predictions and original course materials are excluded from public release. Obtain the authorized local snapshot and verify its exact hash to reproduce this run. An upstream update may require a separately registered data version; do not silently replace this snapshot.

## Intended use

Research, teaching and risk-ranking demonstration only. No production lending, investment, regulatory or annual default-probability claims. Anonymous IDs do not support a real-company matching exercise.
