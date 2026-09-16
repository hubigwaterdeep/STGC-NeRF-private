# Intensity fit/numerical diagnosis — fixed before new measurements

Read-only source: /home/zijiewu/Code/basis4D/log/intensity_coefficients_20260915.
Reuse its established facts; do not rerun preflight, val/test, endpoint replay or
H0–H2 training. Only new diagnostic files and private model copies may change.
No file digests, no publication, no Best/old-results overwrite, no automatic next run.

## Fixed queries and supervision

Official47 legacy train, frames 4950–4953. In each 66×1030 image use full 512-ray
canonical renderer blocks, 768 samples/ray, perturb=False. Query blocks were drawn
once from the 132 complete blocks using numpy.default_rng(700).choice(132,8,False):

| frame | fit block | probe block |
|---|---:|---:|
| 4950 | 32 | 86 |
| 4951 | 60 | 127 |
| 4952 | 29 | 120 |
| 4953 | 44 | 51 |

Pool: 2048 fit rays and 2048 disjoint probe rays. Block selection is independent of
labels, errors and all val/test results. This is a NEW fixed train query cache,
not a claim of reusing the old randomly sampled 1024-ray probe.
Keep all 512×768 query positions and original block context, including unselected
and unsupported samples, for every live T1 bank query. Cache teacher logits,
weights, support, shared geo/direction features, original outputs and coordinates.
Only the frozen T0 bank features may be cached; T1 bank is ALWAYS live and differentiable.

Build original full-frame refiner inputs on these four TRAIN frames only, because
the original U-Net needs image context; extract the fixed query masks without
feeding any corrected intensity to it. Reuse these full-frame passes to measure
range bounds on four train frames; separately report fit/probe range statistics.
This is not a rerun of the old full-validation audit or a claim about all47 frames.

## Numerical read-only paths

1. AMP exactly as source: bank and decoder autocast, correction cast to old FP16
   logit, FP16 addition/sigmoid, FP32 integration on original weights.
2. Same AMP bank/decoder delta but FP32 addition/sigmoid: isolates the final
   addition/activation rounding. No optimizer updates in this path.
3. Private appearance bank/decoder query under autocast disabled, FP32 addition/
   sigmoid/integration. Old cached geometry/logits/weights/refiner stay untouched.
   Frozen tiny-cuda-nn encodings may intrinsically return FP16; their actual dtypes
   are recorded, not relabeled FP32. New spatial tensors and decoder are FP32.

For every path subtract its OWN zero correction output. Report precision-only
zero-path differences separately from learned deltas. Historical final T0/T1 EMA
branches are used only for read-only numerical measurement; fitting starts from
their common historical zero-correction initialization, never the learned endpoint.
Sample strata: all supported, weight>=0.01 (fixed high-weight threshold), TP/FP
supported samples and their high-weight subsets. Ray strata: all, GT-valid, TP, FP,
fit/probe. Statistics: absolute p50/p95/max, signed mean, nonzero ratio/count.
Sigmoid saturation: I<=0.01 or >=0.99; also report exact 0/1 and derivative.
Measure high-order cap hard scales; do not treat observed delta amplitude as a
hard bound. The decoder has no explicit delta clamp or bounded final activation.

## Fixed fit experiment

Two independent private branches and fresh optimizers, common original zero
initialization. Same 17537-parameter decoder. T0 freezes the existing bank;
T1 trains exactly its existing19968 coefficients. Nothing else is trainable.
300 actual Adam updates per arm, lr0.001, eps1e-15, weight_decay0.
LambdaLR 0.1**(step/300), AMP GradScaler init_scale1. No extra objective or sweep.
Report RAW current-parameter fitting curves (no EMA smoothing/selection); this
diagnostic choice is explicit and differs from the historical EMA endpoint report.

Chunk order cycles the four fit blocks, permuted per cycle using NumPy seed19;
each step selects128 distinct fit rays in that block using the same generator.
Both arms use the identical saved schedule. All query positions are fixed, no
new perturbations. Original loss: 0.1 * sum(((J - GT_intensity)*GT_return)^2).
Keep background rays: their loss contribution is exactly0, not an added FP loss.
Log TP/FN positive-return contributions and FP/TN zero contributions separately.
Evaluate entire fit and probe pools at 0/1/10/50/100/300. No tuning or snapshot
selection from probe. Save actual step counts, loss components, gradients and
parameter-relative updates. Record function effects delta_a/delta_I/delta_J too.

Range bound uses actual supported weights A_eff=sum(w*support), alongside looser
sum(w). No delta hard bound exists, so no artificial tighter interval is assumed.
Optimistic independent-sample range projection is not a deployable-model score.

Only fixed-query before/after isolation checks: cached teacher replay parity,
old parameter/buffer immutability, fixed depth/return/refiner channels, and refiner
probabilities on the same four train contexts. No full val/test or original audit.
Stop after the fixed300 updates and diagnosis; no architecture changes or follow-up training.
