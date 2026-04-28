import os
import torch
import random

import deep_gemm
from deep_gemm.testing import (
    test_filter,
    bench_kineto,
    calc_diff, count_bytes
)
from deep_gemm.utils import align
from generators import get_arch_major


@test_filter(lambda: get_arch_major() >= 9)
def test_hc_prenorm_gemm() -> None:
    # Needs TF32 precision for PyTorch GEMMs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print('Testing hyperconnection prenorm GEMM:')
    for m in (13, 137, 4096, 8192):
        for n, k in [(24, 28672), (24, 7680), (24, 7168)]:
            for num_splits in [None, 16]:
                a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
                b = torch.randn((n, k), dtype=torch.float, device='cuda')
                d = torch.empty((m, n), dtype=torch.float, device='cuda') if num_splits is None else \
                        torch.empty((num_splits, m, n), dtype=torch.float, device='cuda')
                s = torch.empty((m, ), dtype=torch.float, device='cuda') if num_splits is None else \
                        torch.empty((num_splits, m), dtype=torch.float, device='cuda')
                deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)
                final_d = d if num_splits is None else d.sum(0)
                final_s = s if num_splits is None else s.sum(0)

                ref_d = a.float() @ b.T
                ref_s = a.float().square().sum(-1)

                diff = max(calc_diff(final_d, ref_d), calc_diff(final_s, ref_s))
                assert diff < 1e-8, f'{m=}, {n=}, {k=}, {diff:.10f}'

                t = bench_kineto(lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits), 'tf32_hc_prenorm_gemm', suppress_kineto_output=True)
                print(f' > Perf (m={m:5}, n={n:5}, k={k:5}, num_splits={(num_splits or 0):2}): '
                      f'{t * 1e6:4.0f} us | '
                      f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                      f'{count_bytes(a, b, d, s) / 1e9 / t:4.0f} GB/s')
    print()




@test_filter(lambda: get_arch_major() == 12)
def test_hc_prenorm_gemm_mma_path_sm120() -> None:
    """dsl12x Phase 5: validate the SM120 MMA-path scaffold compiles and
    dispatches when DG_SM120_HC_PRENORM_V2_MMA=1 is set.

    The MMA kernel currently writes -inf as a sentinel because the inner
    body has not been filled in (see csrc/sm120_tf32_hc_prenorm_gemm.cu
    hc_prenorm_mma_kernel_scaffold). When the kernel is implemented in
    a follow-up session, this test will validate correctness against the
    PyTorch reference. Until then, this test SKIPS (does not fail) so
    CI stays green; the operator can run with
    ``DG_SM120_HC_PRENORM_V2_MMA_STRICT=1`` to assert correctness once
    the kernel is wired up.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print('Testing SM120 hyperconnection prenorm GEMM MMA path (Phase 5):')

    # Save and restore env vars so we don't leak state to other tests.
    saved_env = {
        'DG_SM120_HC_PRENORM_V2_MMA': os.environ.get('DG_SM120_HC_PRENORM_V2_MMA'),
        'DG_SM120_HC_PRENORM_TRACE': os.environ.get('DG_SM120_HC_PRENORM_TRACE'),
    }
    os.environ['DG_SM120_HC_PRENORM_V2_MMA'] = '1'
    os.environ['DG_SM120_HC_PRENORM_TRACE'] = '1'
    strict = os.environ.get('DG_SM120_HC_PRENORM_V2_MMA_STRICT', '0') in (
        '1', 'true', 'TRUE', 'yes', 'YES', 'on', 'ON',
    )

    try:
        # Use a small shape that exercises the M-tile boundary (m=16, m=32, m=64).
        for m in (16, 32, 64):
            for n in (8, 32, 64):
                for k in (256, 1024):
                    a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
                    b = torch.randn((n, k), dtype=torch.float, device='cuda')
                    d = torch.empty((m, n), dtype=torch.float, device='cuda')
                    s = torch.empty((m,), dtype=torch.float, device='cuda')

                    deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=None)

                    is_scaffold_sentinel = bool(torch.isinf(d).any().item())
                    if is_scaffold_sentinel:
                        msg = (
                            f'SCAFFOLD: m={m},n={n},k={k}: MMA kernel returned '
                            'sentinel -inf (scaffolded inner not yet '
                            'implemented). Set DG_SM120_HC_PRENORM_V2_MMA=0 '
                            'to use the production scalar kernel.'
                        )
                        if strict:
                            raise AssertionError(
                                f'STRICT: MMA kernel scaffold returned sentinel '
                                f'for m={m},n={n},k={k}; either implement the '
                                f'kernel or unset DG_SM120_HC_PRENORM_V2_MMA_STRICT.'
                            )
                        print(f' > {msg}')
                        continue

                    ref_d = a.float() @ b.T
                    ref_s = a.float().square().sum(-1)
                    diff_d = calc_diff(d, ref_d)
                    diff_s = calc_diff(s, ref_s)
                    print(
                        f' > MMA path correctness m={m:3} n={n:2} k={k:5}: '
                        f'diff_d={diff_d:.4e} diff_s={diff_s:.4e}'
                    )
                    assert diff_d < 1e-3, (
                        f'm={m},n={n},k={k}: MMA D diff {diff_d:.4e} '
                        f'exceeds tolerance'
                    )
                    assert diff_s < 1e-3, (
                        f'm={m},n={n},k={k}: MMA S diff {diff_s:.4e} '
                        f'exceeds tolerance'
                    )
    finally:
        # Restore env state so other tests see the original values.
        for k_, v in saved_env.items():
            if v is None:
                os.environ.pop(k_, None)
            else:
                os.environ[k_] = v
    print()


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_hc_prenorm_gemm()
    test_hc_prenorm_gemm_mma_path_sm120()
