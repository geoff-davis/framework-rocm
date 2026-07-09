#!/usr/bin/env python3
"""Quick correctness check: is the Framework Desktop GPU usable by PyTorch/ROCm?

Prints ROCm/PyTorch versions and the detected device(s), then verifies a
deterministic matmul result. Pass ``--bench`` to additionally run the attention
(SDPA) benchmark and compare the experimental mem-efficient backend's outputs
and gradients with the math backend. Exits non-zero when compute is unavailable
or produces an invalid result; benchmark failures are also fatal when requested.

To guard against a silent fallback to the wrong GPU/arch, it also checks the
device's arch against EXPECTED_ARCH (default "gfx1151"). A mismatch prints a
warning; set STRICT_ARCH=1 to make it a hard failure, or EXPECTED_ARCH= to skip.
"""
import argparse
import math
import os
import sys
import time
import warnings


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a predictable boolean environment variable."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


def scalar_is_close(actual: float, expected: float, *, rtol: float = 1e-5) -> bool:
    """Return whether a scalar is finite and close to its expected value."""
    return math.isfinite(actual) and math.isclose(actual, expected, rel_tol=rtol)


def optional_positive_float(name: str, default: str = "") -> float | None:
    """Parse an optional positive floating-point environment value."""
    value = os.environ.get(name, default)
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number or empty") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive number or empty")
    return parsed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench",
        action="store_true",
        help="run attention timings and backend numerical checks",
    )
    args = parser.parse_args(argv)

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
    try:
        strict_arch = env_flag("STRICT_ARCH")
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if expected and expected not in arch:
        print(f"WARNING: expected arch '{expected}' not in '{arch}' — wrong GPU "
              "or a fallback may be active (set EXPECTED_ARCH= to silence).",
              file=sys.stderr)
        if strict_arch:
            return 3

    # Exercise the GPU and verify the result, rather than merely checking that
    # a kernel launched without raising.
    dev = torch.device("cuda:0")
    a = torch.ones((1024, 1024), device=dev)
    b = torch.ones((1024, 1024), device=dev)
    c = (a @ b).sum().item()
    torch.cuda.synchronize()
    expected_sum = float(1024**3)
    if not scalar_is_close(c, expected_sum):
        print(
            f"ERROR: GPU matmul returned {c!r}; expected {expected_sum:.1f}",
            file=sys.stderr,
        )
        return 4
    print(f"matmul OK     : sum={c:.3f} on {torch.cuda.get_device_name(0)}")

    if args.bench and not _bench_attention(torch, dev):
        return 5
    return 0


def _relative_l2_error(torch, actual, expected) -> float:
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    denominator = torch.linalg.vector_norm(expected_float).item()
    numerator = torch.linalg.vector_norm(actual_float - expected_float).item()
    return numerator / max(denominator, 1e-12)


def _validate_efficient_attention(torch, F, dev, math_backend, efficient_backend) -> bool:
    """Sanity-check efficient SDPA forward results and gradients against math."""
    from torch.nn.attention import sdpa_kernel

    torch.manual_seed(0)
    shape = (2, 4, 64, 32)
    inputs = tuple(torch.randn(shape, device=dev, dtype=torch.bfloat16) for _ in range(3))

    def evaluate(backend):
        q, k, v = (value.detach().clone().requires_grad_(True) for value in inputs)
        with sdpa_kernel([backend]):
            output = F.scaled_dot_product_attention(q, k, v)
            gradients = torch.autograd.grad(output.float().sum(), (q, k, v))
        torch.cuda.synchronize()
        return output.detach(), tuple(gradient.detach() for gradient in gradients)

    try:
        reference = evaluate(math_backend)
        candidate = evaluate(efficient_backend)
        tensors = (("output", candidate[0], reference[0]),) + tuple(
            (f"grad-{name}", actual, expected)
            for name, actual, expected in zip("qkv", candidate[1], reference[1])
        )
        for label, actual, expected in tensors:
            if not bool(torch.isfinite(actual).all().item()):
                print(f"  ERROR: efficient SDPA {label} contains non-finite values")
                return False
            relative_error = _relative_l2_error(torch, actual, expected)
            print(f"  bf16 {label:7} : relative L2 error={relative_error:.3e}")
            if relative_error > 0.05:
                print(f"  ERROR: efficient SDPA {label} differs materially from math")
                return False
    except Exception as error:  # noqa: BLE001
        print(f"  ERROR: efficient SDPA numerical check failed ({type(error).__name__}: {error})")
        return False
    return True


def _bench_attention(torch, dev) -> bool:
    """Time scaled_dot_product_attention (fwd+bwd) in fp32 vs bf16.

    On gfx1151 the AOTriton mem-efficient SDPA kernel is gated behind
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1; without it SDPA falls back to
    the slow math backend — this surfaces which path is active (and how much
    bf16 recovers), which the matmul check above cannot see. A requested
    benchmark fails if required backends, timings, or numerical checks fail.
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
    aotriton = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "")
    print(f"  aotriton env  : TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL={aotriton or '<unset>'}")
    try:
        require_efficient = env_flag("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
    except ValueError as error:
        print(f"  ERROR: {error}")
        return False

    def run(dtype, iters=10, backend=None):
        # backend: an SDPBackend to pin (raises if its kernel is unavailable
        # for this arch/dtype); None = torch's normal dispatch.
        from contextlib import nullcontext

        try:
            from torch.nn.attention import sdpa_kernel
            ctx = sdpa_kernel([backend]) if backend is not None else nullcontext()
        except ImportError:  # torch < 2.3
            ctx = nullcontext()
        q = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, S, D, device=dev, dtype=dtype, requires_grad=True)
        with ctx, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(3):  # warmup / kernel compile
                F.scaled_dot_product_attention(q, k, v).sum().backward()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                q.grad = k.grad = v.grad = None
                F.scaled_dot_product_attention(q, k, v).sum().backward()
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000.0

    # Default-dispatch timings (what a real workload gets).
    benchmark_ok = True
    for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        try:
            print(f"  {label} fwd+bwd : {run(dtype):8.1f} ms/iter  (default dispatch)")
        except Exception as e:  # noqa: BLE001
            print(f"  {label} fwd+bwd : FAILED ({type(e).__name__}: {e})")
            benchmark_ok = False

    # Pin each backend explicitly so kernel availability (and the math-vs-
    # mem-efficient gap) is visible directly instead of inferred from warnings.
    # On gfx1151 mem-efficient needs TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1.
    mem_efficient_ok = False
    efficient_ms = None
    math_backend = efficient_backend = None
    try:
        from torch.nn.attention import SDPBackend

        math_backend = SDPBackend.MATH
        efficient_backend = SDPBackend.EFFICIENT_ATTENTION
        for label, backend in (
            ("bf16 math-only", math_backend),
            ("bf16 mem-effic.", efficient_backend),
        ):
            try:
                ms = run(torch.bfloat16, backend=backend)
                print(f"  {label} : {ms:8.1f} ms/iter")
                if backend is SDPBackend.EFFICIENT_ATTENTION:
                    mem_efficient_ok = True
                    efficient_ms = ms
            except Exception as e:  # noqa: BLE001
                print(f"  {label} : unavailable ({type(e).__name__})")
                if backend is math_backend or require_efficient:
                    benchmark_ok = False
    except ImportError:
        print("  (torch < 2.3: per-backend probe unavailable)")
        if require_efficient:
            benchmark_ok = False

    if not mem_efficient_ok:
        print(
            "  NOTE: mem-efficient SDPA is not running -> math fallback (bf16 "
            "~11x slower here, and the SxS materialization OOMs training jobs). "
            "Set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 (run.sh/compose "
            "default it on) — verified faster on gfx1151 with torch 2.10 / "
            "ROCm 7.2.4; the 2026-07-02 'slower' finding was the older stack. "
            "See docs/gfx1151-attention-findings.md."
        )
    elif math_backend is not None and efficient_backend is not None:
        benchmark_ok = (
            _validate_efficient_attention(torch, F, dev, math_backend, efficient_backend)
            and benchmark_ok
        )
        try:
            max_efficient_ms = optional_positive_float(
                "MAX_BF16_EFFICIENT_MS", "25" if require_efficient else ""
            )
        except ValueError as error:
            print(f"  ERROR: {error}")
            benchmark_ok = False
        else:
            if (
                max_efficient_ms is not None
                and efficient_ms is not None
                and efficient_ms > max_efficient_ms
            ):
                print(
                    f"  ERROR: efficient bf16 SDPA took {efficient_ms:.1f} ms; "
                    f"limit is {max_efficient_ms:.1f} ms"
                )
                benchmark_ok = False
    return benchmark_ok


if __name__ == "__main__":
    raise SystemExit(main())
