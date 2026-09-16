# Actual commands executed (2026-09-16, Australia/Perth)

Each phase was run once, sequentially, after confirming no other GPU compute job.
Do not rerun these commands into the existing output; create a NEW diagnostic copy
if independently reproducing. No command invokes val/test or the old launchers.

```bash
cd /home/zijiewu/Code/STGC-NeRF-private/reports/logs/intensity_fit_diagnosis_20260916
source /home/zijiewu/Code/basis4D/log/local_setup_20260915/environment.sh
python -m py_compile diagnostic_core.py build_cache.py
python -u build_cache.py > cache.log 2>&1
python -u measure_numeric.py > numeric.log 2>&1
python -m py_compile run_fit.py measure_numeric.py
python -u run_fit.py > fit.log 2>&1
python summarize_diagnosis.py
```

The source snapshot is a filesystem copy of
`/home/zijiewu/Code/basis4D/log/intensity_coefficients_20260915/source`,
excluding `__pycache__`, preserving dependency symlinks. Original source is not
edited. Only diagnostic instrumentation substitutes an in-memory QueryView class
to record high-order cap scales, returning the unchanged original computation.

Environment: original activation script and virtualenv, PyTorch2.7.1+cu128,
CUDA12.8, RTX5090. All original checkpoint tensors are loaded with map_location=cpu
and then into a private frozen model. Old correction_ema.pth is loaded ONLY into
a private branch for read-only measurement. Fit copies instead load the historical
`initial_correction.pth`, each with a fresh optimizer; diagnostic weights are not
published or promoted to Best.

Pre-execution choices: CONFIGURATION.md. Actual split: data_split.json.
All300 per-arm fit selections: schedule.json. Actual updates, losses, gradients
and relative parameter changes: fit/T0 and fit/T1 optimizer_updates.jsonl.
Dataset is inherited official47/legacy; full manifest/pose/normalization metadata
is in inherited_data_alignment.json. That file records historical val/test metadata
as provenance only; no val/test dataset was loaded or queried by this diagnosis.

No SHA or replacement file digest was computed/verified. Historical result and
model files were read only. All new files reside under this independent directory.
