# SA2 Kernel Improvement Plan — In-Kernel Threshold Block Skipping (INT8 QK + Block Skip + FP8 PV)

- Created: 2026-09-30
- Status: Plan (not implemented; all items pending Owner approval)
- Target: `D:\USERFILES\GitHub\SageAttention` (Owner's Windows fork, incl. SageAttention3 Blackwell dir) + HSWQ loader integration
- Goal: fuse threshold-based KV-block skipping directly into the SA2 attention kernel — the "INT8 QK + block skip + FP8 PV" combination that neither SageAttention upstream nor NVIDIA ModelOpt ships
- Method: every claim below verified against the actual fork source on this machine (file + line references included). Hypotheses that need measurement are labeled as such.

---

## 0. Verified baseline (fork source, read this turn)

### 0.1 The Triton kernel already contains a skip scaffold

`sageattention/triton/attn_qk_int8_per_block.py::_attn_fwd_inner` (L20-70):

```python
for start_n in range(lo, hi, BLOCK_N):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    mask_block = None
    skip = False
    if mask_ptrs is not None:
        if mask_ptrs.dtype.element_ty == tl.int1:
            mask_block = tl.load(mask_ptrs + start_n * stride_maskn, ...)
            if tl.max(mask_block) == 0:
                skip = True                    # <-- block-skip hook EXISTS
    if not skip:
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)   # INT8 dot (cheap)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        ...
        v = tl.load(V_ptrs, ...)               # V load (skip target)
        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)  # PV MMA (skip target = expensive side)
    K_ptrs += BLOCK_N * stride_kn
    K_scale_ptr += 1                           # per-KV-block scale: usable as skip-decision input
    V_ptrs += BLOCK_N * stride_vn
```

Key structural facts:
- `skip` flag already short-circuits the entire body (K/V loads, QK dot, softmax, PV accumulate)
- `K_scale_ptr` advances one entry per KV block -> **per-block amax scales are available in-loop**
- The loop iterates KV in `BLOCK_N` blocks (64 in the per-block path)

### 0.2 CUDA kernel (the hot path on this machine)

`csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh` — `qk_int_sv_f8_attn_kernel` template:

- Main KV loop: `for (uint32_t iter = 1; iter < num_iterations - 1; iter++)` with cp.async double-buffering (K/V staged one iteration ahead)
- Per-iteration flow: `compute_int_qk` (INT8 MMA) -> int->float RS conversion -> `update_mdo` (running max update) -> `accumulate_d`/`accumulate_d_f8` (denominator) -> `RS_32_to_8` (P->FP8) -> `compute_fp8_sv_inst_buf[_fp16_accu]` (PV MMA) -> stage next K/V
- Per-block scale consumed in-loop: `dequant_scale = q_scale * K_scale[k_scale_idx + iter * k_scale_advance_offset]` — K scales are per-KV-block
- The fork builds with `TORCH_CUDA_ARCH_LIST` from `torch.cuda.get_arch_list()` (README / build scripts), so SM120 (RTX 5060 Ti) native compilation is in scope; the Triton kernel serves as the semantics reference

### 0.3 Dispatch on this machine

- `sageattention/core.py::sageattn` dispatches by arch/availability (SM89/SM90 compiled modules + Triton fallback)
- HSWQ calls `sageattn` through `hswq/hswq_sa2_accel.py::_make_attention_sage2` (single integration point; checkpoint-probe pattern resolution, per-call stats and timing already in place)
- **Phase 0 must audit which entry (compiled CUDA vs Triton) actually serves the calls on SM120**, so kernel mods land on the hot path

### 0.4 Prior-plan context

- HSWQ SA2 plan (`md/2026-09-10_sa2_attention_acceleration_plan.md` in the loader repo): SA3 fully rejected (FP4 attention error accumulates across layers); SA2 validated through Phase 1
- Profile (1024x1024, 6 steps, RTX 5060 Ti): attention 26.4% of step time; rotate 16.9%; cuBLAS FP4 GEMM 19.6%; quantize 1.8%

---

## 1. Core design: two-tier threshold block skipping

### 1.1 Tier 1 — post-QK, pre-PV skip (mathematically exact, no calibration)

**Rule**: after the INT8 `qk` dot is computed and `block_max = max(qk)` is known, skip the block's PV/normalization contribution when

```
block_max <= m_i - T          (T >= ~17 in exp2 domain for fp16-level safety)
```

- **Why this is exact, not approximate**: online softmax with running max `m_i`. `exp2(score - m_i) <= 2^-17 ~ 7.6e-6` per element for skipped blocks; a full BLOCK_N=64..128 block contributes < 1e-3 relative to `l_i >= 1`. If a later block raises `m_i`, `alpha = exp2(m_i - m_ij)` only shrinks earlier contributions — a skip decision against the current max can never become wrong later. Holds for causal and non-causal alike.
- **What it saves**: `exp2` for the block, P->FP8 pack (`RS_32_to_8`), PV MMA (`compute_fp8_sv_inst_buf*`), V smem pipeline traffic, and the denominator update — the expensive side. The INT8 QK dot stays (cheap side; it produces the skip decision).
- **Decision cost**: a tile-local max reduction — register-local, negligible vs one PV MMA.
- **Knob**: `T` as kernel template/arg (`SKIP_THRESHOLD_LOG2`); 0 = feature off.

### 1.2 Tier 2 — pre-QK skip via calibrated statistics (optional, only after Tier 1 proves out)

- **Idea**: static diffusion workloads (fixed resolution/step schedule) have stable per-(layer, step-bucket, KV-block) score statistics across seeds/steps. A small side-table lets the kernel skip the ENTIRE block (including INT8 dot and K smem staging).
- **Calibration harness source (ModelOpt, reuse the harness only)**: `sparsity/attention_sparsity/calibration/calibrator.py::DynamicThresholdCalibrator`, `plugins/sparse_attn_calibration.py::split_records_by_phase / merge_count_records / fit_from_counts`. Re-granularize from ModelOpt's 128-token tiles to SA2's BLOCK_N. Do NOT reuse ModelOpt's attention backend/kernels.
- **Safety**: Tier-2 skip must be backed by a runtime margin check (block max vs threshold with headroom); on any violation fall through to Tier-1 logic for that block. Tier 2 accelerates the decision; it never bypasses the math without margin.
- **Recommendation**: v1 = Tier 1 only. Measure before building statistics infrastructure.

### 1.3 Explicitly out of scope

- ModelOpt skip-softmax runtime (FP16 FlashAttention path — superseded for HSWQ by this plan)
- SpargeAttn (same fusion concept; if evaluated later, benchmark against this implementation, never stacked)
- Token merging (ToMe-class; different axis)

---

## 2. Implementation plan

### Phase 0 — dispatch audit + offline skip-ceiling measurement (no kernel change)

1. Log the dispatched backend per call on SM120 (env-gated debug in `core.py::sageattn`).
2. Through the HSWQ SA2 wrapper, capture per-layer stats for one 12-step SDXL run: M per call, KV block count, per-call ms, and the offline Tier-1 skip ceiling (fraction of blocks whose max score sat below running max minus T, computed from score dumps on representative steps).
3. Go/no-go gate: if the estimated skipped-PV fraction across self-attention layers is below ~15% of PV work, stop here — do not implement on hope (SDXL self-attn entropy is the open question).

### Phase 1 — Triton kernel (reference implementation)

1. `_attn_fwd_inner`: after `m_ij`, compute the tile-wide block max; skip the PV/`l_i`/`m_i` update when `block_max <= m_i - T` (tile-uniform branch; V/K pointers still advance). `SKIP_THRESHOLD_LOG2` as constexpr arg; 0 disables.
2. Correctness gate: random-tensor comparison vs unmodified kernel (fp16 tolerance), then the standard 1024x1024/12-step SSIM protocol, same seeds — acceptance: SSIM delta within noise vs plain SA2.
3. Speed gate: per-call ms on self-attention layers; report realized skip ratio.

### Phase 2 — CUDA kernel (hot path)

1. Port the rule into the `iter` loop of `qk_int_sv_f8_cuda_sm89.cuh`: after `RS_f32` and `update_mdo`, take the per-tile max; if the CTA-uniform skip condition holds, skip `RS_32_to_8`, `accumulate_d*`, and `compute_fp8_sv_inst_buf*` for that iteration. cp.async pipeline accounting must be preserved (staged K/V consumed or predicated off without breaking `cp_async::wait_group<1>` invariants — settle against the double-buffer design).
2. Template flag (e.g., `bool enable_block_skip`) so a non-skip build remains for A/B; keep the fork's minimal-patch policy (additive, gated, off by default).
3. Rebuild with the fork's build scripts (arch list from torch).
4. Gates: same two-level correctness + speed gates as Phase 1.

### Phase 3 — HSWQ integration

1. `core.py`: plumb `skip_threshold_log2` (default 0) through `sageattn_qk_int8_pv_fp8_cuda` and the Triton entry.
2. `hswq/hswq_sa2_accel.py::_make_attention_sage2`: pass per-layer threshold from env or a JSON side-file (loader conf style); pattern cross-check machinery untouched.
3. Stats: extend SA2 counters with `skipped_blocks / total_blocks`.
4. Document the knob + default in the fork README; bump build metadata.

### Phase 4 (optional) — Tier 2 calibration

1. Offline: dump per-(layer, step, block) max-score records across seeds (wrapper-side hook on the Triton reference kernel).
2. Fit skip tables with the ModelOpt-derived harness (BLOCK_N granularity).
3. Kernel consumes the table (pre-QK skip with runtime margin check).
4. Tighter SSIM gate than Tier 1 (this stage is an approximation).

---

## 3. Risks and mitigations

| Risk | Reality check | Mitigation |
|---|---|---|
| SDXL self-attn not peaked enough -> few skips | Unverified; prior data does not cover skip ratio | Phase 0 offline ceiling measurement; hard go/no-go gate |
| Branch divergence kills CUDA win | Decision is CTA-uniform (whole CTA_K block), no intra-warp divergence | Tile/CTA-level decision only, never per-thread |
| cp.async pipeline accounting broken by skip | K/V staged one iteration ahead | Settle in Phase 2 against wait_group invariants; Triton path unconstrained |
| fp16 accumulator interaction | Skipping reduces fp32+fp16 flush cycles; `d` normalization unchanged | Random-tensor + SSIM gates |
| T too aggressive | exp2-domain T=17 ~ 7.6e-6/element, below fp16 epsilon | T exposed as arg; conservative default |
| Fork divergence from upstream | Owner policy: minimal patch set | Additive + gated off by default; one template flag + one pybind arg |

## 4. Success criteria

- Phase 1/2 gates: SSIM delta within noise vs plain SA2 (12-step, 1024x1024, same seeds); per-step ms improvement > 0 with realized skip ratio reported
- End-to-end bar: attention stage (26.4% baseline) reduced by the realized skipped-PV fraction; overall step-time reduction >= 3% to keep the feature ON by default (otherwise ship gated OFF)

## 5. Relation to the HSWQ plan (loader repo, `md/2026-09-30_hswq_speed_vram_optimization_plan.md`)

- This plan SUPERSEDES item C4. C4's caution applied to adopting ModelOpt's FP16 skip-softmax runtime; this plan instead moves the skip into SA2's quantized kernel, where "INT8 QK + skip + FP8 PV" actually exists as one kernel.
- HSWQ A-series items (Krea2 CUDA Graph, scale freeze default) are unaffected and complementary: graphs are per-Linear, skip is per-attention-call; different modules.

## 6. File map (verified this turn)

| File | Role |
|---|---|
| `sageattention/triton/attn_qk_int8_per_block.py` | Triton reference kernel; `_attn_fwd_inner` skip scaffold (L20-70) |
| `sageattention/triton/quant_per_block.py` | per-block INT8 quant kernel; block scales origin |
| `csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh` | main CUDA kernel template; `iter` loop = Phase 2 hook |
| `csrc/qattn/attn_utils.cuh` | device fns: `update_mdo`, `accumulate_d[_f8]`, `RS_32_to_8`, `compute_fp8_sv_inst_buf[_fp16_accu]`, masks |
| `csrc/qattn/sm89_qk_int8_sv_f8_accum_f16_attn_inst_buf.cu` | launcher; signature to extend with threshold arg |
| `csrc/qattn/pybind_sm89.cpp` | pybind arg addition |
| `sageattention/core.py` | dispatch + new-arg plumbing (938 L) |
| `sageattention3_blackwell/` | SA3 dir — the REJECTED FP4 path; untouched |
| HSWQ: `hswq/hswq_sa2_accel.py` | `_make_attention_sage2` integration point + stats |
