# STGC: basis/latent representation audit and next implementation plan

## Scope and status

This document proposes new experiments. No STGC training, checkpoint evaluation, repository write, or model promotion was performed for this review.

Audited remote versions:
- Standard source: `main`, commit `154a200e21a8b7b96e599d79e56c8fc93712cf3e`.
- Basis source snapshot: `codex/diagnosis-20260916`, commit `21899bcc6cbb498146b25b0d168b30392a78f484`, under `reports/logs/intensity_fit_diagnosis_20260916/source/`.

The recent local multi-scene experiment source has not been established as byte-identical to this snapshot. Confirm the relevant class names, mathematical operations, active options and initialization policy before treating the audit as an explanation of those results. Do not calculate file hashes merely for this check.

## Main findings

1. The basis field is already learned. Modal arrays/hash tables, analytic frequencies/envelopes/gains, adaptive spline parameters and high-order mixing are trainable. Direct coefficient optimization is learning; it differs from generating coefficients through shared network weights.
2. Standard STGC is not a basis-free latent baseline. `HashGridT.forward` interpolates eight time-indexed hash encoders, while `interpT` applies four Lagrange basis weights to feature channels.
3. At a fixed spatial position, one standard dynamic hash projection/level has the form `sum_j hat_j(t) sum_b L_b(t) c_jb(x)`. There are 32 nominal coefficient functions. The attached CPU algebra script obtains numerical rank 29 for its design matrix on 51 uniform times. This is illustrative, not the actual data protocol or a trained-model result.
4. The audited modal hash instead stores eight coefficients per spatial lookup/level. In the high-order branch, the same coefficients are contracted with the mean and the gated high-order temporal bases. With the RMS cap inactive, its form is `sum_r c_r(x) [phi_r(t) + g_r psi_r(t)]`. This is at most an eight-dimensional temporal span for that isolated branch and a fixed learned dictionary. It is NOT the rank of the entire renderer.
5. A network `C(x) = D(z(x), x)` changes spatial sharing and optimization, but does not enlarge that temporal span by itself. Independent high-order coefficient fields can increase the available temporal span; unstructured local temporal latent channels offer a different way to retain temporal flexibility.
6. The default basis wrapper selects `anchored_spline_high_order_geometry_residual`. The historical field constructor freezes its parent, but the from-scratch wrapper explicitly re-enables training. Do not diagnose a frozen-trunk bug from the parent constructor alone.
7. This model is more than a basis-only replacement. It includes learned fusion with tanh/LayerNorm and a geometry residual. In specialized density queries, density uses full features but appearance consumes the base geometry feature. Keep these choices constant in a representation comparison.
8. The high-order RMS limiter depends on the query tensor. It can make output context-dependent when active. Read the already-completed QUERY_CONSISTENCY report before repeating this diagnostic; code inspection does not establish it as the source of observed degradation.

## Research pointers

These works support task-specific design choices, not a universal theorem that bases outperform latent representations.

- NeX, CVPR 2021: https://nex-mpi.github.io/ ; https://github.com/nex-mpi/nex-code
- NeuRBF, ICCV 2023: https://arxiv.org/html/2309.15426 ; https://github.com/oppo-us-research/NeuRBF
- Dictionary Fields, SIGGRAPH 2023 journal track: https://apchenstu.github.io/FactorFields/ ; https://github.com/autonomousvision/factor-fields
- DynMF, ECCV 2024: https://arxiv.org/html/2312.00112 ; https://github.com/agelosk/dynmf
  The official repository inspected in this review still advertises code as forthcoming. Do not assume a complete training implementation is available there.
- Neural Parametric Gaussians, CVPR 2024: https://arxiv.org/html/2312.01196 ; https://github.com/DevikalyanDas/npgs

## Proposed first experiment: partial basis / local temporal-latent hybrid

Question: Did replacing all dynamic temporal hash levels with the shared rank-eight representation discard useful local temporal flexibility?

Start with the hash branch only. Do not simultaneously alter planes, flow, the renderer, task weights, refiner, fusion, or geometry residual.

- Existing control: all eight dynamic hash levels use the current modal representation.
- New candidate: keep levels 0-3 modal; use standard STGC time-indexed hash/interpT for levels 4-7.
- This 4/4 allocation is a prespecified engineering starting point, not a literature-derived optimum.
- Preserve each level's actual spatial resolution. Splitting an encoder must not silently recompute different per-level scale factors.
- Preserve the three spatial projection roles, one scalar output per level, 24 dynamic hash outputs and the overall 120-dimensional field interface.
- Preserve the original normalized time protocol and actual time-indexed latent interpolation. Call this a local temporal-latent route, not a basis-free representation.
- Account for all hash table values and network parameters. Report an exact-capacity-matched control where feasible; otherwise report the capacity difference explicitly and avoid attributing gains solely to representation.
- No new post-render depth/intensity residual and no canonical warp.
- Keep the retained modal time basis formulas, high-order refiner, gates and temporal training policy intact. They remain trainable under the existing from-scratch policy.
- The new field is a primary representation trained end-to-end from scratch. Do not turn this into another frozen-field intensity-correction experiment.

This is the first new architectural experiment to implement. Do not automatically launch the later candidates below.

## Later controlled candidates

### Independent low/high spatial coefficients

For retained modal levels, compare tied and independent coefficients:

`tied: h(x,t) = sum_r c_r(x)[phi_r(t)+g_r psi_r(t)]`

`untied: h(x,t) = sum_r c_low_r(x)phi_r(t) + sum_r c_high_r(x)g_r psi_r(t)`

Allocate the additional coefficient capacity from a registered representation budget. A larger model is not a clean test of sharing. Keep temporal atoms/gating/aggregation unchanged. This isolates spatial coefficient tying, not neural generation.

### Neural coefficient generation

Only after an explicit untied control exists, replace a selected coefficient field with:

`z(x) = interpolate(compact_spatial_latent_table, x)`

`c_high(x) = D_theta(z(x), spatial_encoding(x), role_embedding, level_embedding)`

Keep `c_low(x)` explicitly optimized and keep static features/local free temporal channels as latent representations. Both explicit coefficients and latent codes remain trainable.

Use a small decoder and count its parameters, latent tables, runtime and memory. Preserve the coordinate domain of each role in the first version. Feeding full xyz into a previously one-/two-dimensional factor is another structural change and needs a separate control.

Do not feed arbitrary time into the coefficient generator in this experiment. That would change the time factorization itself and confound the intended comparison. Nonlinear decoding after interpolation is not equivalent to interpolating predecoded coefficients; choose one order, document it and test it.

This network introduces spatial sharing; it does not by itself guarantee additional temporal rank or cross-scene generalization. Cross-scene priors require a corresponding training/evaluation protocol.

## Minimum code tests and required reports

- Confirm the active source classes and training parameter groups without changing Best/lastBest.
- Verify temporal endpoints, exact time knots, interpolation shape/order and existing normalization conventions.
- Verify the all-modal setting reproduces the old forward path under the same numerical conventions.
- Verify every selected trainable table/network receives gradients, without detaching or caching a trainable feature path.
- Compare current/next/previous query semantics with the original flow-based aggregation; do not silently change the no_grad policy.
- Verify feature dimensions, source coordinate domains and per-level resolutions.
- Report parameter count, peak memory, training step time and ray throughput.
- Use fixed seeds, frame splits, sampled-ray schedules and fixed training budgets. Keep raw/EMA and pre-/post-refiner evaluation conventions separate.

For an initial diagnosis, use previously identified failing scenes 8120/10200 and a favorable scene 3353. They are diagnostic examples, not a replacement for the full evaluation set. Use development data for decisions; do not repeatedly inspect test results to select models. If these scenes were already examined as tests, describe the new study as post-hoc development and retain a genuinely untouched evaluation for final claims.

Report CD, F-score, depth RMSE, intensity RMSE and return metrics together, including train/development and moving/static/edge/range strata when reliable labels are available. Do not declare success from CD alone. Do not combine the hypothesis with AdaTask, Recon, new losses, extra temporal modes or a new refiner in the same run.

Stop after the registered first candidate/control budget and write the decision. No automatic experiment queue, full benchmark sweep or model promotion. If required local assets are unavailable, deliver the implementation/tests and clearly mark real training as BLOCKED, rather than retraining the baseline implicitly.

## Scope of the algebra script

`check_stgc_temporal_span.py` requires NumPy only. It reconstructs the temporal design matrix implied by the source, verifies its partition-of-unity identities, and prints its singular spectrum. The rank-8/16 entries are mathematical upper bounds under the assumptions in the output, not measured trained-basis ranks. Neither the script nor this review proves the cause of the benchmark regressions.
