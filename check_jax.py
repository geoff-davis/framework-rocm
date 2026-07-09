#!/usr/bin/env python3
"""Quick correctness check: is the Framework Desktop GPU usable by JAX/ROCm?

Prints JAX/jaxlib versions and the detected devices, then verifies a
deterministic matmul result. Pass ``--bench`` to additionally run an attention
micro-benchmark (fwd+bwd, fp32 vs bf16) and validate that its outputs and
gradients are finite. Exits non-zero if no GPU-backed device is usable or a
requested check fails.

JAX doesn't expose the gfx arch, so to catch a silent fallback to the wrong GPU
it checks the device's reported kind against EXPECTED_DEVICE (default
"Radeon 80", matching any Strix Halo variant — 8060S, 8050S). A mismatch prints
a warning; set STRICT_DEVICE=1 to make it a hard failure, or EXPECTED_DEVICE=
to skip.
"""
import argparse
import math
import os
import sys
import time


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench",
        action="store_true",
        help="run attention timings and numerical sanity checks",
    )
    args = parser.parse_args(argv)

    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("ERROR: jax is not installed in this environment.", file=sys.stderr)
        return 2

    print(f"jax version   : {jax.__version__}")
    try:
        import jaxlib
        print(f"jaxlib version: {jaxlib.__version__}")
    except Exception:
        pass

    devices = jax.devices()
    print(f"devices       : {devices}")

    # On ROCm, JAX reports the GPU backend as platform 'gpu' (ROCm/HIP under
    # the hood). If the only device is CPU, the GPU backend didn't load.
    gpu_devices = [d for d in devices if d.platform == "gpu"]
    if not gpu_devices:
        print("ERROR: no GPU device visible to JAX (only CPU backend loaded).",
              file=sys.stderr)
        print("Check device passthrough (--device=/dev/kfd --device=/dev/dri), "
              "render/video group membership, and that the base image's JAX "
              "includes gfx1151 (see README for the AMD gfx1151 wheel fallback).",
              file=sys.stderr)
        return 1

    dev = gpu_devices[0]
    kind = getattr(dev, "device_kind", "") or ""
    print(f"device kind   : {kind}")
    expected = os.environ.get("EXPECTED_DEVICE", "Radeon 80")
    try:
        strict_device = env_flag("STRICT_DEVICE")
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if expected and expected not in kind:
        print(f"WARNING: expected device '{expected}' not in '{kind}' — wrong GPU "
              "or a fallback may be active (set EXPECTED_DEVICE= to silence).",
              file=sys.stderr)
        if strict_device:
            return 3

    # Actually exercise the GPU.
    a = jax.device_put(jnp.ones((1024, 1024)), dev)
    b = jax.device_put(jnp.ones((1024, 1024)), dev)
    c = (a @ b).sum().block_until_ready()
    result = float(c)
    expected_sum = float(1024**3)
    if not scalar_is_close(result, expected_sum):
        print(
            f"ERROR: GPU matmul returned {result!r}; expected {expected_sum:.1f}",
            file=sys.stderr,
        )
        return 4
    print(f"matmul OK     : sum={result:.1f} on {dev}")

    if args.bench and not _bench_attention(jax, jnp, dev):
        return 5
    return 0


def _bench_attention(jax, jnp, dev) -> bool:
    """Time jax.nn.dot_product_attention (fwd+bwd) in fp32 vs bf16.

    The explicit ``cudnn`` probe reports whether that implementation is
    available, but does not infer how XLA compiles or fuses its default path.
    """
    try:
        from jax.nn import dot_product_attention
    except ImportError:
        print("\nattention bench: jax.nn.dot_product_attention unavailable")
        return False

    B, H, S, D = 32, 12, 512, 64  # ~bge-base attention shape; layout is BSHD
    print(f"\nattention fwd+bwd micro-benchmark  [B={B} H={H} S={S} D={D}]")

    # Probe one explicit implementation. implementation=None lets XLA choose;
    # rejection of 'cudnn' does not imply that XLA's default graph is unfused.
    def make_qkv(dtype):
        key = jax.random.PRNGKey(0)
        kq, kk, kv = jax.random.split(key, 3)
        shape = (B, S, H, D)
        return (jax.device_put(jax.random.normal(kq, shape, dtype), dev),
                jax.device_put(jax.random.normal(kk, shape, dtype), dev),
                jax.device_put(jax.random.normal(kv, shape, dtype), dev))

    try:
        q, k, v = make_qkv(jnp.float32)
        dot_product_attention(q, k, v, implementation="cudnn").block_until_ready()
        print("  fused kernel  : 'cudnn' implementation ACCEPTED")
    except Exception as e:  # noqa: BLE001
        print(f"  cudnn path    : unavailable ({type(e).__name__}); benchmarking XLA default")

    def run(dtype, iters=10):
        q, k, v = make_qkv(dtype)

        def loss(q, k, v):
            return dot_product_attention(q, k, v).sum()

        grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
        for _ in range(3):  # warmup / XLA compile
            g = grad_fn(q, k, v)
        jax.block_until_ready(g)
        output = dot_product_attention(q, k, v).block_until_ready()
        if not bool(jnp.all(jnp.isfinite(output))):
            raise FloatingPointError("attention output contains non-finite values")
        for gradient in jax.tree.leaves(g):
            if not bool(jnp.all(jnp.isfinite(gradient))):
                raise FloatingPointError("attention gradient contains non-finite values")
        t0 = time.perf_counter()
        for _ in range(iters):
            g = grad_fn(q, k, v)
        jax.block_until_ready(g)
        return (time.perf_counter() - t0) / iters * 1000.0

    benchmark_ok = True
    for label, dtype in (("fp32", jnp.float32), ("bf16", jnp.bfloat16)):
        try:
            ms = run(dtype)
            print(f"  {label} fwd+bwd : {ms:8.1f} ms/iter")
        except Exception as e:  # noqa: BLE001
            print(f"  {label} fwd+bwd : FAILED ({type(e).__name__}: {e})")
            benchmark_ok = False
    return benchmark_ok


if __name__ == "__main__":
    raise SystemExit(main())
