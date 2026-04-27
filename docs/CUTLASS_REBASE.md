# CUTLASS rebase guide for SM120 (Phase 4 prep)

## Why rebase

The build currently uses the CUTLASS pinned in `third-party/cutlass/`. To
unlock Phase 4 (dense FP8xFP8 K=64 tile variant + correct NVFP4 MoE), the
build must include:

1. **PR #3121** (NVIDIA/cutlass) — fixes K=64 blockscaled tiles for
   SM120/SM121. Without this, K=64 produces wrong results, which is the
   reason the current `csrc/sm120_fp8_fp8_cutlass.cu` only uses
   `TileShape = Shape<_128, _128, _128>`.
2. **FlashInfer NVFP4 MoE autotuner-tactic patches** — without these, NVFP4
   MoE on SM120 was producing garbage output. They are upstreamed via
   FlashInfer issue #2847 / #2851 patch series.
3. **`compute_120f` kernel-schedule support** — needed for the cooperative
   variant of `KernelPtrArrayTmaWarpSpecialized*` on CUDA 13.

## Choosing a commit

Pin to a CUTLASS commit that:
- Is on `main` after PR #3121's merge.
- Contains the FlashInfer NVFP4 fix for SM120 (look for changes to
  `cutlass/gemm/collective/sm120_blockscaled_*.hpp` after FlashInfer #2851).
- Compiles cleanly under CUDA 13.0+.

Recommended approach:

```bash
cd third-party/cutlass
git fetch --tags --all
# Pick a recent tag that includes #3121:
git checkout v4.0.0   # or newer
cd ../..
git add third-party/cutlass
git commit -m "Pin CUTLASS to v4.0.0+ (PR #3121, NVFP4 MoE fix)"
```

If the project is using a submodule rather than a vendored tree, update with
`git submodule update --remote` and verify the pinned hash includes #3121.

## Verifying the rebase

After rebase, rebuild the extension:

```bash
docker compose exec -T vllm bash -lc \
  'cd /workspace/DeepGEMM && DG_FORCE_BUILD=1 MAX_JOBS=8 \
   python3 setup.py build_ext --inplace 2>&1 | tail -50'
```

Then run the regression gate:

```bash
docker compose exec -T vllm bash -lc \
  'cd /workspace/DeepGEMM && scripts/regression_gate.sh --label phase4-rebase'
```

If the regression gate is green and there is no NaN in dense MoE output for a
representative chat completion, the rebase is safe to commit.

## Phase 4.1: enabling the K=64 dense FP8 variant

Once the rebase is in place, the K=64 variant in `csrc/sm120_fp8_fp8_cutlass.cu`
can be enabled. Implementation outline:

1. Add a second template instantiation alongside the existing K=128 one:
   ```cpp
   using TileShapeK64 = Shape<_128, _128, _64>;
   using ClusterShape = Shape<_1, _1, _1>;
   // ... CollectiveBuilder/CollectiveEpilogue specialized on TileShapeK64 ...
   ```
2. Gate selection in the host dispatcher behind
   `DG_SM120_DENSE_K64=1` and a narrow-M shape predicate (e.g., `m <= 4 &&
   k % 64 == 0 && (n in {1024, 1536, 2048, 4096})`).
3. Add a microbenchmark `scripts/bench_sm120_dense_k64.py` to compare
   K=64 vs K=128 for the dense FP8 projection shapes used by V4 Flash.
4. After acceptance, default-on for the small-M shape predicate.

This change is **deliberately deferred** until after the rebase because PR
#3121 is the precondition for K=64 correctness on SM120.

## Phase 4.2: NVFP4 MoE cooperative schedule

After the rebase, `KernelPtrArrayTmaWarpSpecializedCooperative` becomes
available with `compute_120f`. Add it as an opt-in alongside the current
`KernelPtrArrayTmaWarpSpecializedPingpong` and benchmark on m in {6,12,24}.

The current MoE chain budget at MTP-shape `m=12,n=4096,k=2048` is FC1
~96 us / FC2 ~53 us. The cooperative schedule's win on consumer Blackwell is
typically larger than ping-pong's only when `m * (n / TileN)` is high enough
to fill the SM. For DeepSeek V4 Flash MTP it is borderline; benchmark before
flipping.

## Phase 4.3: autotuner skipped tiles

Per FlashInfer issue #2847, the CUTLASS autotuner skips
`(128,128,64)`, `(128,256,64)`, and `(256,128,64)` on SM120. Manually
instantiate these and benchmark for V4 Flash MoE shapes; if any beats the
autotuner's pick, hard-pin in `csrc/sm120_fp8_fp4_cutlass.cu`.

## Risk register

| Risk                                                  | Mitigation                                                                |
| ----------------------------------------------------- | ------------------------------------------------------------------------- |
| CUTLASS rebase breaks existing FP8 paths              | Phase 0 archives baseline; revert if regression gate fails                |
| K=64 path still fails despite PR #3121                | Keep K=128 as default; gate K=64 narrowly                                 |
| NVFP4 MoE regression on rebase                        | Run end-to-end correctness (tiny chat returning `OK.`); revert if needed  |
| Submodule vs vendored confusion                       | Document the pinned hash in this file after rebase                        |
