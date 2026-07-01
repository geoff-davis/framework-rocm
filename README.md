# framework-rocm

A reproducible ROCm + PyTorch container for the **Framework Desktop** — the AMD
Ryzen AI Max ("Strix Halo") board with the integrated **Radeon 8060S** GPU
(`gfx1151`). It exists because getting a recent ROCm and a matching PyTorch
talking to this GPU is fiddly, and re-deriving the right versions every time is
a waste of an afternoon.

Built on AMD's official [`rocm/pytorch`](https://hub.docker.com/r/rocm/pytorch)
image so ROCm and PyTorch come pre-pinned to a known-good pair; this repo adds
the device passthrough, the gfx1151 specifics, and a smoke test.

## Requirements (host)

- A Framework Desktop (Ryzen AI Max / Strix Halo) — or any gfx1151 machine.
- A recent kernel with `amdgpu` loaded (`/dev/kfd` and `/dev/dri/renderD*` present).
- Docker with the Compose plugin.
- Your user in the `render` and `video` groups on the host:
  ```bash
  sudo usermod -aG render,video "$USER"   # then log out/in
  ```

## Quick start

With Compose:

```bash
docker compose build
docker compose run --rm rocm python /usr/local/bin/check_gpu.py   # smoke test
docker compose run --rm rocm                                      # interactive shell
```

Or with the plain-`docker` wrapper:

```bash
./run.sh build
./run.sh check      # smoke test
./run.sh shell      # interactive shell
./run.sh python your_script.py
```

A successful smoke test looks roughly like:

```
torch version : 2.6.0+rocm6.4.1
ROCm/HIP ver  : 6.4.xxxxx
device count  : 1
  [0] AMD Radeon Graphics
matmul OK     : sum=… on AMD Radeon Graphics
```

## What makes the GPU visible

These aren't optional decorations — they're why compute works inside the
container:

| Flag | Why |
| --- | --- |
| `--device=/dev/kfd` | ROCm's kernel fusion driver (compute). |
| `--device=/dev/dri` | GPU render nodes. |
| `--group-add render` / `video` | Own `/dev/kfd` and `/dev/dri/render*`. |
| `--security-opt seccomp=unconfined` | ROCm userspace trips the default seccomp profile. |
| `--ipc=host` | Avoids shared-memory limits for larger tensors / dataloaders. |

If names don't resolve to the right GIDs inside the container, find them on the
host with `getent group render video` and use the numeric IDs.

## Picking a ROCm / PyTorch version

The image tag is a build arg — `ROCM_PYTORCH_TAG` — defaulting to a version
known to work on gfx1151. To move it:

1. Browse tags at https://hub.docker.com/r/rocm/pytorch/tags and pick one whose
   ROCm version lists gfx1151 (Strix Halo / RDNA3.5) support.
2. Build against it:
   ```bash
   docker compose build --build-arg ROCM_PYTORCH_TAG=<tag>
   # or:  ROCM_PYTORCH_TAG=<tag> ./run.sh build
   ```
3. Re-run the smoke test.

## The gfx1151 gotcha

Recent ROCm supports gfx1151 natively, but some libraries/kernels are only
fully tuned for nearby archs and can throw `invalid device function` or
missing-kernel errors. If that happens, present the GPU as `gfx1100` (RDNA3
desktop) by setting:

```
HSA_OVERRIDE_GFX_VERSION=11.0.0
```

There's a commented line for it in both the `Dockerfile` and `compose.yaml`.
**Try without it first** — the override can mask real problems and cost
performance. Only enable it if the native path genuinely fails.

## Layout

```
Dockerfile         # builds on rocm/pytorch, adds gfx1151 env + smoke test
compose.yaml       # device passthrough + groups + seccomp, ready to run
run.sh             # same, for the plain-docker path (build/shell/check/…)
requirements.txt   # extra Python deps baked into the image (keep minimal)
check_gpu.py       # smoke test: versions, device list, real GPU matmul
workspace/         # bind-mounted into /workspace (git-ignored)
```

## Adding Python packages

Put extra deps in `requirements.txt` and rebuild. **Don't** add a bare `torch`
— that pulls a CUDA/CPU wheel and clobbers the ROCm build already in the base
image.

## License

MIT — see [LICENSE](LICENSE).
