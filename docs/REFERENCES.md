# References checked during implementation

- scikit-learn, leakage and preprocessing: https://scikit-learn.org/stable/common_pitfalls.html
- scikit-learn, Average Precision (non-interpolated weighted recall increments): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- scikit-learn, probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- imbalanced-learn, resampling pitfalls: https://imbalanced-learn.org/stable/common_pitfalls.html
- Gorishniy et al. (2021), Revisiting Deep Learning Models for Tabular Data: https://arxiv.org/abs/2106.11959
- Authors' official tabular-model repository: https://github.com/yandex-research/rtdl-revisiting-models
- Candidate upstream dataset, README and citation requests: https://github.com/sowide/bankruptcy_dataset
- Current upstream license (CC BY 4.0): https://github.com/sowide/bankruptcy_dataset/blob/main/LICENSE.md

The installed package versions, not the current website version, are recorded in `results/public/environment.json` and the lock files. Reference links explain algorithm/protocol choices; they do not validate the local label as a forward default event. The local snapshot and live upstream comparison are recorded by hash. PyTorch loss uses BCEWithLogitsLoss and evaluation uses sigmoid; installed-runtime behavior is exercised by software tests. No FT-Transformer paper performance is claimed for this compact implementation.
