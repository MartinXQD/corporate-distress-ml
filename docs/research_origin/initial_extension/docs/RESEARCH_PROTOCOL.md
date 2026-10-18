# Registered research protocol

Owner: Qidian (Martin) Xing. Independent extension.

The immutable machine specification is `configs/protocol_v1.yaml`; its registration hash is in `configs/protocol_registration.json`. H1 compares S0/S1, H2 S1/S2, H3 model families, H4 selection and sampling, and H5 company novelty/stress.

The task is retrospective failed-company classification. Financial-statement publication dates, true bankruptcy event dates and a fixed event horizon are unavailable. All historical years have previously been inspected. The 2015-2018 evaluation cannot be described as a new untouched holdout.

D1 trains through 2005 and scores 2006-2007; D2 through 2007 scores 2008-2009; D3 through 2009 scores 2010-2011; D4 through 2011 scores 2012. Select by equal-weight mean AP over these seven scoring years. Early stopping uses only the last training year, followed by full-window refitting at fixed length.

Final base-model fit: 1999-2012. Calibration: 2013. Threshold: 2014. Historical evaluation: 2015-2018, once after model/config/artifact freeze. Group stress uses three fixed SHA256 groups and pre-registered untuned recipes; held-out companies cannot enter any fit, calibration or stopping role.

All data-dependent transforms and selectors are refit within their training role. No company ID, year or lifetime summary is a model feature. Dynamic windows use exact calendar offsets. All target rows are retained. Model artifacts and row-level predictions remain local.

Training is bounded by 6 hours soft, 8 hours hard including delivery, and 600 fit calls including internal selection and stopping refits. Every fit starts with a durable ledger event. Two threads, one neural training process. No paid resources.
