# Model card

Independent extension by Qidian (Martin) Xing, with disclosed Codex assistance.

## Task and selection

Retrospective company-constant failed-label classification. No verified annual event horizon or statement-publication dates. The final ML representative is Random Forest (S1); the final DL representative is Residual MLP (S1). Selection uses equal-weight AP across seven development years. The historical period was previously inspected.

## Measured seed-42 historical performance

ML: AP 0.205641, ROC-AUC 0.861870, macro annual recall at Top5% 0.459578. DL: AP 0.105345, ROC-AUC 0.815105, macro recall at Top5% 0.294140. Same-protocol S0 Logistic AP: 0.153109. The paired 95% AP-difference interval for selected RF versus Logistic is [-0.001797, 0.101701], crossing zero; positive improvement is inconclusive. The selected ROS network and fixed ensemble failed to transfer development gains. See the full table rather than treating these representatives as test-selected winners.

## Validation and uncertainty

Final fit through 2012, Platt 2013, threshold 2014, historical evaluation 2015-2018. All final representatives use 5 registered seeds, not best-seed selection. Paired company bootstrap uses 2,000 draws, conditional on fitted models. Unseen-positive company count through 2012 is only 16. Untuned three-group company stress excludes each held-out company from all fitting and calibration.

## Artifacts and operation

Each local final artifact includes preprocessing, selected and transformed feature order, weights/estimator, calibration, threshold, budget rule, history window, configuration, fit logs and hashes. Numerical reload tolerance is 1e-6 on the measured CPU environment. Dynamic models require prior calendar inputs; rank ensembles require an annual cohort. Do not load untrusted source pickles or mutate the freeze for a repeat historical search.

## Limits and intended use

Research/teaching/risk-ranking demonstration only. Scores are not validated one-year PDs or causal explanations. Unknown units, label timing, historical test exposure, anonymous company identity, prevalence shift, repeated observations, small unseen-positive cohorts and macro shocks limit transportability. No production, lending, investment or profit claim is supported.
