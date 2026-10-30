# SpargeAttn Evaluation & Improvement Plan — with NVIDIA ModelOpt (bit-width preserving)

- Created: 2026-09-30 (v2: rewritten under the corrected premise — see Revision history)
- Status: Plan (not implemented; all items pending Owner approval)
- Target: SpargeAttn (`D:\USERFILES\GitHub\SpargeAttn`, CUDA build) evaluated in the HSWQ loader, improved with techniques from NVIDIA Model Optimizer
- Method: every claim verified against source on this machine; hypotheses labeled as needing measurement

## Revision history

| Version | Content |
|---|---|
| v1 | Initial plan (contained a proposal to convert QK INT8→NVFP4) |
| v2 | **QK NVFP4 proposal RETRACTED.** Owner identified it as the same structure as SA3 (FP4 attention error accumulates across layers; NVFP4+SA3 cos 0.0556 = rejected). v2 premise: **no quantization bit-width changes** — QK stays INT8, PV stays FP8. All improvements are bit-width-preserving. |

---

## 0. Premise (v2, hard constraints)

1. **QK stays INT8** (256-level granularity). Converting QK to FP4 repeats SA3's failure mode; the skip decision itself depends on QK score fidelity.
2. **PV stays FP8** (E4M3, SageAttention2 path). PV→FP4 is on hold pending quality evidence; SA3 is the cautionary precedent.
3. Improvement surface = **calibration accuracy, scale selection, kernel schedule, and integration** — never precision reduction.
4. Success gates unchanged: SSIM within noise vs plain SA2 (1024×1024, 12 steps, same seeds), s/it improvement, skip ratio reported.

---

## 1. Phase E1 — Build & evaluate stock SpargeAttn (CUDA)

1. Build: `cd D:\USERFILES\GitHub\SpargeAttn && pip install ninja && python setup.py install`. CUDA >= 12.8 for Blackwell (RTX 5060 Ti = SM120, supported).
2. Integration test: call `spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=0.5)` standalone; verify kernel launches on SM120 (`bfd980b` fixed sm89- launch errors; verify SM120 coverage in build log).
3. HSWQ integration: swap `sageattn` inside `hswq/hswq_sa2_accel.py::_make_attention_sage2` (few-line evaluation patch; pattern cross-check machinery untouched).
4. Measure (1024×1024, 12 steps, same seeds): SSIM vs plain SA2, s/it, skip ratio (`return_sparsity=True`).
5. Gates: SSIM within noise AND s/it improvement → adopt; otherwise keep plain SA2 and stop here.

## 2. Phase E2 — bit-width-preserving improvements from ModelOpt

All items below keep INT8 QK / FP8 PV. None reduce precision.

### E2-a. Better scale selection for Q/K (improves skip-decision fidelity and score accuracy)

- ModelOpt `kernels/quantization/gemm/nvfp4_fp8_scale_sweep.py` + `_fp8_scale_candidates.py`: one Triton kernel evaluates N scale candidates per tensor (`_fp8_scale_sweep_kernel`), plus a Hessian-weighted variant (`_fp8_scale_sweep_hessian`).
- Adaptation: the same sweep structure applied to **INT8 per-block scale selection** in SpargeAttn's `quant_per_block.py` — generate candidates (current max-based scale ± offsets, outlier-clipped scales), score each against the fp16 reference block output inside one kernel, pick min-error. No bit-width change; only which scale is chosen.
- Effect: lower QK quantization error → sparser true score distribution → higher real skip ratio at the same quality, and less risk of the first-stage filter misjudging borderline blocks.
- Note: this is an offline/first-call selection; per-block structure stays identical.

### E2-b. Activation headroom for the skip threshold (from ModelOpt NVFP4 act-headroom concept)

- ModelOpt `quantization/calib/nvfp4_act_headroom.py`: adds headroom to activation scales so downstream quantization never saturates.
- Adaptation (bit-width-preserving): calibrate a small headroom into the **skip-threshold margins** (τ/θ/λ per layer) and into `K_scale` handling — i.e., calibrate margins on real activation statistics so borderline blocks that look skippable on average but occasionally spike are NOT skipped. Reduces rare outlier-induced flicker artifacts.
- Mechanism: offline calibration pass over representative prompts (HSWQ already runs sensitivity analysis; same data), producing per-layer threshold offsets stored beside the τ/θ/λ config.

### E2-c. Tied/shared-weight scale consistency (QKV-fused layers)

- ModelOpt `quantization/utils/shared_input.py::find_shared_input_groups` + `SharedWeightGlobalAmaxState`: keeps scales consistent across weight-sharing modules.
- Adaptation: in models where Q/K/V come from a fused projection (or share weights), ensure K per-block scales used by the skip filter are computed from the same statistics as the fused weight's, avoiding inconsistent skip masks between consumers of the same block.
- Relevant when HSWQ packs contain fused QKV (SDXL does not; some DiTs do). Verify per model before applying.

### E2-d. TopCdf prediction refinement (algorithmic, no precision change)

- ModelOpt's Hessian-weighted sweep idea (weight the error by importance) translates to SpargeAttn's first-stage prediction: weight the compressed attention map's TopCdf selection by block importance (HSWQ's DualMonitor sensitivity data), so borderline blocks in high-importance layers are computed even if predicted skippable.
- Implementation: per-layer importance scalar multiplies τ (effective threshold) — a config-level change in `get_block_map_meansim` call parameters, no kernel change.
- Effect: concentrates compute on sensitive layers; measured SSIM/speed tradeoff must be reported.

## 3. Explicitly out of scope (v2)

- QK NVFP4 / FP4 anywhere in the skip-decision path (SA3 structure — rejected by Owner)
- PV FP4 conversion (held pending quality evidence; same precedent)
- ModelOpt skip-softmax runtime, token merging, any precision reduction

## 4. Order of execution

| # | Item | Type | Size | Risk |
|---|---|---|---|---|
| 1 | E1 build + stock evaluation | infra | M | low |
| 2 | E2-b threshold headroom calibration | accuracy | S | low |
| 3 | E2-a INT8 scale sweep | accuracy/speed | M | low |
| 4 | E2-d importance-weighted τ | quality/speed | S | low |
| 5 | E2-c shared-scale consistency | accuracy | S | low (per-model check) |

## 5. Validation protocol

- SSIM: 1024×1024 / 12 steps / same seeds vs plain SA2 baseline.
- Speed: s/it via existing bench nodes; report skip ratio per layer (`return_sparsity`).
- Calibration data: HSWQ's existing DualMonitor sensitivity outputs; no new datasets.
- Separation: loader integration stays inside `hswq_sa2_accel.py` evaluation path; no shared-patch edits; SpargeAttn repo changes limited to build scripts if needed.
