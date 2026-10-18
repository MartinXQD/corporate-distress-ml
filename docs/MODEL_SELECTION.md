# Why the frozen rank blend is the official model

The official model follows the recorded development decision. It is not the best model chosen after observing historical results.

The [selection freeze](../configs/selection_freeze.json) was written at 2026-10-18T00:04:05Z. It selects XGBoost recipe `refine_r3_15_a028aa7a` and MLP recipe `refine_r3_21_bbf84fb1` using 2009–2013 development years and three registered seeds. The historical period had already been exposed in earlier research; the procedural freeze does not turn it into an untouched test set.

| XGBoost weight | MLP weight | Development mean annual AP |
| ---: | ---: | ---: |
| 0.00 | 1.00 | 0.538859 |
| 0.25 | 0.75 | 0.573697 |
| **0.50** | **0.50** | **0.581580** |
| 0.75 | 0.25 | 0.580505 |
| 1.00 | 0.00 | 0.551330 |

The finite nonnegative grid and admission rule required at least +0.005 annual AP over the reference, with ROC and review-recall guardrails. The equal blend met that rule and was frozen before the one-time 2014 confirmation and final historical scoring. [Exact development results](../results/ensemble_development.csv)

On 2015–2018, the five-seed MLP representative has mean annual AP 0.392916, while the frozen blend has 0.373320. This reversal is reported openly. MLP alone is a useful retrospective comparator; promoting it after reading those scores would change the recorded selection policy. The blend's pooled ROC-AUC is 0.921602, while its mean annual ROC-AUC is 0.926730; these are different metrics.

All four strong baselines share the final 2014 cutoff and registered seeds. The lag-feature XGBoost baseline has annual AP 0.302835. Thus the primary annual AP difference is +0.070485, rather than a comparison with the earlier 0.206 pooled-AP result under different roles.
