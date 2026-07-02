#!/usr/bin/env python3
"""Quick smoke test: is the Framework Desktop GPU visible to PyTorch/ROCm?

Prints ROCm/PyTorch versions and the detected device(s), runs a tiny matmul to
confirm compute works, then runs an **attention (SDPA) micro-benchmark** in
fp32 vs bf16. The attention timing matters: on gfx1151 there is currently no
flash/mem-efficient SDPA kernel, so attention silently falls back to a slow
math backend — a regression a matmul-only check sails right past (see
docs/gfx1151-attention-findings.md). Exits non-zero if the GPU isn't usable.

To guard against a silent fallback to the wrong GPU/arch, it also checks the
device's arch against EXPECTED_ARCH (default "gfx1151"). A mismatch prints a
warning; set STRICT_ARCH=1 to make it a hard failure, or EXPECTED_ARCH= to skip.
Set SKIP_ATTENTION_BENCH=1 to skip the attention timing.
"""
import os
import sys
import time
import warnings


def main() -> int:
    try:
        import torch
    except ImportError:
        print("ERROR: torch is not installed in this environment.", file=sys.stderr)
        return 2

    print(f"torch version : {torch.__version__}")
    print(f"ROCm/HIP ver  : {getattr(torch.version, 'hip', None)}")

    if not torch.cuda.is_available():
        # On ROCm builds, torch.cuda is the HIP backend.
        print("ERROR: no GPU visible to PyTorch (torch.cuda.is_available() is False).",
              file=sys.stderr)
        print("Check device passthrough (--device=/dev/kfd --device=/dev/dri) and "
              "that your user is in the render/video groups on the host.",
              file=sys.stderr)
        return 1

    count = torch.cuda.device_count()
    print(f"device count  : {count}")
    for i in range(count):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or ""
    print(f"gpu arch      : {arch}")
    expected = os.environ.get("EXPECTED_ARCH", "gfx1151")
    if expected and expected not in arch:
        print(f"WARNING: expected arch '{expected}' not in '{arch}' — wrong GPU "
              "or a fallback may be active (set EXPECTED_ARCH= to silence).",
              file=sys.stderr)
        if os.environ.get("STRICT_ARCH"):
            return 3

    # Actually exercise the GPU.
    dev = torch.device("cuda:0")
    a = torch.randn(1024, 1024, device=dev)
    b = torch.randn(1024, 1024, device=dev)
    c = (a @ b).sum().item()
    torch.cuda.synchronize()
    print(f"matmul OK     : sum={c:.3f} on {torch.cuda.get_device_name(0)}")

    if not os.environ.get("SKIP_ATTENTION_BENCH"):
        _bench_attention(torch, dev)
    return 0


def _bench_attention(torch, dev) -> None:
    """Time scaled_dot_product_attention (fwd+bwd) in fp32 vs bf16.

    On gfx1151 there is no flash/mem-efficient SDPA kernel, so SDPA falls back
    to the slow math backend — this surfaces that (and how much bf16 recovers),
    which the matmul check above cannot see. Informational: never fails the smoke.
    """
    import torch.nn.functional as F

    B, H, S, D = 32, 12, 512, 64  # ~bge-base attention shape
    print(f"\nattention (SDPA) fwd+bwd micro-benchmark  [B={B} H={H} S={S} D={D}]")

    try:
        cb = torch.backends.cuda
        print(
            f"  sdpa flags    : flash={cb.flash_sdp_enabled()} "
            f"mem_efficient={cb.mem_efficient_sdp_enabled()} math={cb.math_sdp_enabled()} "
            "(a 'flag on' does NOT mean the kernel exists for this arch)"
        )
    except Exception:  # noqa: BLE001
        pass

    def run(dtype, iters=10):
        q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
        for _ in range(3):  # warmup / kernel compile
            F.scaled_dot_product_attention(q, k, v).sum().backward()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            q.grad = k.grad = v.grad = None
            F.scaled_dot_product_attention(q, k, v).sum().backward()
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000.0

    fell_back = False
    for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ms = run(dtype)
            if any("attention" in str(w.message).lower() for w in caught):
                fell_back = True
            print(f"  {label} fwd+bwd : {ms:8.1f} ms/iter")
        except Exception as e:  # noqa: BLE001
            print(f"  {label} fwd+bwd : FAILED ({type(e).__name__}: {e})")
    if fell_back:
        print(
            "  NOTE: torch reports mem-efficient/flash attention is unavailable for "
            "this GPU -> slow math fallback. Use bf16 (+ gradient checkpointing) for "
            "training; do NOT set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 (it is "
            "slower on gfx1151). See docs/gfx1151-attention-findings.md."
        )


if __name__ == "__main__":
    raise SystemExit(main())
