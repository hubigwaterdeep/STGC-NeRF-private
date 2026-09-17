# Modi01 experiment registration

From scratch; official legacy frame IDs. No file digests are computed.
This is post-hoc development; val/test overlap and cannot establish untouched-test performance.
Source: `/home/zijiewu/Code/STGC-NeRF-private/reports/logs/intensity_fit_diagnosis_20260916/source`

Frame IDs, normalization and manifests: `/home/zijiewu/Code/STGC-NeRF-private/Modi01/reports/preflight_3353/registration.json`

Fixed seed 0; maximum 30000 optimizer updates.
Use identical budgets, seed, input ordering, ray sampler and patch schedule for controls.
Report all five metric families together; label raw/EMA and pre/post-refiner separately.
Moving/static/edge/range strata require reliable labels; otherwise report unavailable.
Stop after this single arm. No automatic experiment queue, benchmark sweep or promotion.
