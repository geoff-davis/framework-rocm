#!/usr/bin/env python3
"""Quick smoke test: is the Framework Desktop GPU visible to PyTorch/ROCm?

Prints ROCm/PyTorch versions and the detected device(s), then runs a tiny
matmul on the GPU to confirm compute actually works — not just that the device
is enumerated. Exits non-zero if the GPU isn't usable.
"""
import sys


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

    # Actually exercise the GPU.
    dev = torch.device("cuda:0")
    a = torch.randn(1024, 1024, device=dev)
    b = torch.randn(1024, 1024, device=dev)
    c = (a @ b).sum().item()
    torch.cuda.synchronize()
    print(f"matmul OK     : sum={c:.3f} on {torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
