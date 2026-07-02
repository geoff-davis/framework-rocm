#!/usr/bin/env python3
"""Quick smoke test: is the Framework Desktop GPU visible to JAX/ROCm?

Prints JAX/jaxlib versions and the detected devices, runs a tiny matmul to
confirm compute works, then runs an **attention micro-benchmark** (fwd+bwd,
fp32 vs bf16) and probes whether any fused attention implementation exists.
gfx1151 has no flash/mem-efficient attention kernel in torch (see
docs/gfx1151-attention-findings.md); this surfaces the equivalent situation on
the JAX side, which a matmul-only check cannot see. Exits non-zero if no
GPU-backed device is usable.

JAX doesn't expose the gfx arch, so to catch a silent fallback to the wrong GPU
it checks the device's reported kind against EXPECTED_DEVICE (default
"Radeon 80", matching any Strix Halo variant — 8060S, 8050S). A mismatch prints
a warning; set STRICT_DEVICE=1 to make it a hard failure, or EXPECTED_DEVICE=
to skip. Set SKIP_ATTENTION_BENCH=1 to skip the attention timing.
"""
import os
import sys
import time


def main() -> int:
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
    if expected and expected not in kind:
        print(f"WARNING: expected device '{expected}' not in '{kind}' — wrong GPU "
              "or a fallback may be active (set EXPECTED_DEVICE= to silence).",
              file=sys.stderr)
        if os.environ.get("STRICT_DEVICE"):
            return 3

    # Actually exercise the GPU.
    a = jax.device_put(jnp.ones((1024, 1024)), dev)
    b = jax.device_put(jnp.ones((1024, 1024)), dev)
    c = (a @ b).sum().block_until_ready()
    print(f"matmul OK     : sum={float(c):.1f} on {dev}")

    if not os.environ.get("SKIP_ATTENTION_BENCH"):
        _bench_attention(jax, jnp, dev)
    return 0


def _bench_attention(jax, jnp, dev) -> None:
    """Time jax.nn.dot_product_attention (fwd+bwd) in fp32 vs bf16.

    Mirrors check_gpu.py's SDPA benchmark: gfx1151 has no fused attention
    kernel in torch, and this surfaces whether JAX/XLA is in the same boat —
    plus whether any fused implementation ('cudnn') is claimed to exist.
    Informational: never fails the smoke test.
    """
    try:
        from jax.nn import dot_product_attention
    except ImportError:
        print("\nattention bench: jax.nn.dot_product_attention unavailable — skipped")
        return

    B, H, S, D = 32, 12, 512, 64  # ~bge-base attention shape; layout is BSHD
    print(f"\nattention fwd+bwd micro-benchmark  [B={B} H={H} S={S} D={D}]")

    # Probe for a fused implementation. implementation=None lets XLA choose;
    # 'cudnn' (the fused path on NVIDIA, mapped to hipDNN/none on ROCm) tells
    # us whether a flash-style kernel is even claimed for this stack.
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
        print(f"  fused kernel  : none ('cudnn' rejected: {type(e).__name__}) -> XLA math path")

    def run(dtype, iters=10):
        q, k, v = make_qkv(dtype)

        def loss(q, k, v):
            return dot_product_attention(q, k, v).sum()

        grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
        for _ in range(3):  # warmup / XLA compile
            g = grad_fn(q, k, v)
        jax.block_until_ready(g)
        t0 = time.time()
        for _ in range(iters):
            g = grad_fn(q, k, v)
        jax.block_until_ready(g)
        return (time.time() - t0) / iters * 1000.0

    for label, dtype in (("fp32", jnp.float32), ("bf16", jnp.bfloat16)):
        try:
            ms = run(dtype)
            print(f"  {label} fwd+bwd : {ms:8.1f} ms/iter")
        except Exception as e:  # noqa: BLE001
            print(f"  {label} fwd+bwd : FAILED ({type(e).__name__}: {e})")


if __name__ == "__main__":
    raise SystemExit(main())
