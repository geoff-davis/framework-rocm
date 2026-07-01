#!/usr/bin/env python3
"""Quick smoke test: is the Framework Desktop GPU visible to JAX/ROCm?

Prints JAX/jaxlib versions and the detected devices, then runs a tiny matmul on
the GPU to confirm compute actually works — not just that a device is
enumerated. Exits non-zero if no GPU-backed device is usable.
"""
import sys


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

    # Actually exercise the GPU.
    dev = gpu_devices[0]
    a = jax.device_put(jnp.ones((1024, 1024)), dev)
    b = jax.device_put(jnp.ones((1024, 1024)), dev)
    c = (a @ b).sum().block_until_ready()
    print(f"matmul OK     : sum={float(c):.1f} on {dev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
