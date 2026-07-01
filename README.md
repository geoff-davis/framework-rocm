# framework-rocm

Reproducible ROCm containers for the **Framework Desktop** — the AMD Ryzen AI
Max ("Strix Halo") board with the integrated **Radeon 8060S** GPU (`gfx1151`).
It exists because getting a recent ROCm and a matching PyTorch or JAX talking to
this GPU is fiddly, and re-deriving the right versions every time is a waste of
an afternoon.

Two sibling images, each built on the matching official AMD base so ROCm and the
framework come pre-pinned to a combination AMD tested:

- **`pytorch`** — from [`rocm/pytorch`](https://hub.docker.com/r/rocm/pytorch)
- **`jax`** — from [`rocm/jax`](https://hub.docker.com/r/rocm/jax)

They're kept separate on purpose: each framework bundles its own ROCm userspace
libs, so separate images avoid version friction and keep pulls smaller. Both
share the same device passthrough, gfx1151 specifics, and smoke-test pattern.

> **gfx1151 support, honestly:** AMD's official
> [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
> only lists the Ryzen AI Max+ 395 / Radeon 8060S (gfx1151) as *officially*
> supported through **ROCm 6.4.4**. In practice, **ROCm 7.2.x works — and works
> better** (updated HSA runtime, refreshed `amdgpu` module, broader kernel
> coverage), and the community runs real PyTorch workloads on it. This repo
> defaults to 7.2.4 for that reason. If you want to stay strictly on the
> officially-supported stack, pin a 6.4.4 tag instead (see below).

## Requirements (host)

- A Framework Desktop (Ryzen AI Max / Strix Halo) — or any gfx1151 machine.
- A recent kernel with `amdgpu` loaded (`/dev/kfd` and `/dev/dri/renderD*` present).
- Docker with the Compose plugin.
- Your user in the `render` and `video` groups on the host:
  ```bash
  sudo usermod -aG render,video "$USER"   # then log out/in
  ```

## Quick start

Pick a framework — `pytorch` or `jax`. With Compose:

```bash
docker compose build pytorch                                        # or: jax
docker compose run --rm pytorch python /usr/local/bin/check_gpu.py  # smoke test
docker compose run --rm jax     python /usr/local/bin/check_jax.py  # smoke test
docker compose run --rm pytorch                                     # interactive shell
```

Or with the plain-`docker` wrapper (first arg is the framework):

```bash
./run.sh pytorch build
./run.sh pytorch check          # smoke test
./run.sh jax     check          # smoke test
./run.sh pytorch shell          # interactive shell
./run.sh jax     python your_script.py
```

A successful PyTorch smoke test looks roughly like:

```
torch version : 2.10.0+rocm7.2.4
ROCm/HIP ver  : 7.2.xxxxx
device count  : 1
  [0] AMD Radeon Graphics
matmul OK     : sum=… on AMD Radeon Graphics
```

…and the JAX one:

```
jax version   : 0.8.2
jaxlib version: 0.8.2
devices       : [RocmDevice(id=0)]
matmul OK     : sum=1048576.0 on RocmDevice(id=0)
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

## Picking versions

Each image's base tag is a build arg. Browse tags, build against one, re-run the
smoke test.

**PyTorch** — `ROCM_PYTORCH_TAG`, default
`rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`
([tags](https://hub.docker.com/r/rocm/pytorch/tags)):

```bash
docker compose build --build-arg ROCM_PYTORCH_TAG=<tag> pytorch
# or:  ROCM_PYTORCH_TAG=<tag> ./run.sh pytorch build
```

- Default is newest / best real-world Strix Halo behaviour.
- Older PyTorch on the same ROCm: swap `_2.10.0` for `_2.9.1` / `_2.8.0` / `_2.7.1`.
- Officially-supported stack: a `rocm6.4.4_*` tag, matching AMD's matrix.

**JAX** — `ROCM_JAX_TAG`, default `rocm7.2.4-jax0.8.2-py3.12`
([tags](https://hub.docker.com/r/rocm/jax/tags)):

```bash
docker compose build --build-arg ROCM_JAX_TAG=<tag> jax
# or:  ROCM_JAX_TAG=<tag> ./run.sh jax build
```

Keep the JAX / jaxlib / `jax-rocm7-plugin` versions aligned with the ROCm major
— that mismatch is the usual JAX-on-ROCm failure mode. AMD's `rocm/jax` tags
already bundle a matched set.

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

**If the image's bundled PyTorch doesn't see gfx1151 at all**, the robust fix
is to reinstall PyTorch from AMD's gfx1151-aware wheel index (the stock
pytorch.org wheels don't include gfx1151 kernels). Add to the `Dockerfile`
after the `FROM`, matching the index URL to your ROCm version:

For **PyTorch**, reinstall from AMD's gfx1151-aware wheel index (add to
`Dockerfile` after the `FROM`, matching the index URL to your ROCm version):

```dockerfile
RUN pip install --no-cache-dir --force-reinstall \
    torch torchvision \
    --index-url https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
```

For **JAX**, AMD publishes gfx1151-specific wheels (JAX uses a PJRT plugin, so
`jax` itself is pure Python from PyPI + the ROCm backend wheels). Add to
`Dockerfile.jax`:

```dockerfile
RUN pip install --no-cache-dir --force-reinstall \
    jax jaxlib jax-rocm7-plugin jax-rocm7-pjrt \
    --extra-index-url https://repo.amd.com/rocm/whl/gfx1151/
```

Run the smoke test after — if it already passes on the base image, you don't
need this.

## Layout

```
Dockerfile           # PyTorch image: FROM rocm/pytorch + gfx1151 env
Dockerfile.jax       # JAX image:     FROM rocm/jax     + gfx1151 env
compose.yaml         # two services sharing GPU passthrough (YAML anchor)
run.sh               # plain-docker path: ./run.sh {pytorch|jax} {build|shell|check|…}
requirements.txt     # extra deps for the PyTorch image (keep minimal)
requirements-jax.txt # extra deps for the JAX image (keep minimal)
check_gpu.py         # PyTorch smoke test: versions, devices, real GPU matmul
check_jax.py         # JAX smoke test: versions, devices, real GPU matmul
workspace/           # bind-mounted into /workspace (git-ignored)
```

## Adding Python packages

Put extra deps in `requirements.txt` (PyTorch) or `requirements-jax.txt` (JAX)
and rebuild. **Don't** re-add the framework itself — a bare `torch` pulls a
CUDA/CPU wheel, and adding `jax`/`jaxlib`/`jax-rocm7-*` risks clobbering the
ROCm-matched build already in the base image.

## License

MIT — see [LICENSE](LICENSE).
