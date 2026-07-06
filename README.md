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
>
> Both images here were **verified on an actual Framework Desktop**: the ROCm
> 7.2.4 `rocm/pytorch` and `rocm/jax` bases both see gfx1151 (Radeon 8060S) and
> run GPU compute *natively* — no `HSA_OVERRIDE_GFX_VERSION` and no gfx1151
> fallback wheels required.

## Requirements (host)

- A Framework Desktop (Ryzen AI Max / Strix Halo) — or any gfx1151 machine.
- A recent kernel with `amdgpu` loaded (`/dev/kfd` and `/dev/dri/renderD*` present).
- Docker with the Compose plugin.
- Your user in the `render` and `video` groups on the host:
  ```bash
  sudo usermod -aG render,video "$USER"   # then log out/in
  ```

## Quick start

Pick a framework — `pytorch` or `jax`.

**With Compose**, first create a `.env` (host-specific GIDs + your UID/GID —
the compose path needs it; `run.sh` does not):

```bash
cp .env.example .env      # then edit, or just generate it:
mkdir -p ~/.cache/framework-rocm && \
printf 'RENDER_GID=%s\nVIDEO_GID=%s\nHOST_UID=%s\nHOST_GID=%s\n' \
  "$(getent group render | cut -d: -f3)" "$(getent group video | cut -d: -f3)" \
  "$(id -u)" "$(id -g)" > .env
```

Then:

```bash
docker compose build pytorch                                        # or: jax
docker compose run --rm pytorch python /usr/local/bin/check_gpu.py  # smoke test
docker compose run --rm jax     python /usr/local/bin/check_jax.py  # smoke test
docker compose run --rm pytorch                                     # interactive shell
```

Or with the plain-`docker` wrapper (first arg is the framework; no `.env`
needed — it resolves the GIDs and maps your user automatically):

```bash
./run.sh pytorch build
./run.sh pytorch check          # smoke test
./run.sh jax     check          # smoke test
./run.sh pytorch shell          # interactive shell
./run.sh jax     python your_script.py
```

A successful PyTorch smoke test looks roughly like:

```
torch version : 2.10.0+rocm7.2.4.git3d3aa833
ROCm/HIP ver  : 7.2.53211
device count  : 1
  [0] Radeon 8060S Graphics
gpu arch      : gfx1151
matmul OK     : sum=… on Radeon 8060S Graphics
```

…and the JAX one:

```
jax version   : 0.8.2
jaxlib version: 0.8.2+rocm7.2.4
devices       : [RocmDevice(id=0)]
device kind   : Radeon 8060S Graphics
matmul OK     : sum=1073741824.0 on rocm:0
```

Both tests verify the device matches expectations, so a silent fallback to the
wrong GPU shows up instead of passing quietly: PyTorch checks the arch string
(`gfx1151`), and JAX — which doesn't expose the arch — checks the device kind
against `Radeon 80` (matching any Strix Halo variant: 8060S, 8050S). On a
different card, set `EXPECTED_ARCH=` / `EXPECTED_DEVICE=` to silence the warning,
or `STRICT_ARCH=1` / `STRICT_DEVICE=1` to make a mismatch a hard failure.

### Runtime behaviour (both paths)

- **Runs as your host user** so files written to the mounted workspace aren't
  root-owned. Need root (e.g. `pip install` into system site-packages)?
  `ROCM_ROOT=1 ./run.sh pytorch shell`, or `HOST_UID=0`/`HOST_GID=0` in `.env`.
- **`HOME` is `/workspace`** inside the container (the mapped user has no
  passwd entry, so it would otherwise be homeless and `$HOME`-writing tools
  would break).
- **Caches persist across runs**: a host dir (default `~/.cache/framework-rocm`,
  override with `ROCM_CACHE_DIR`) is mounted at `$HOME/.cache`, so pip
  downloads and MIOpen's compiled-kernel cache survive container exit. MIOpen
  especially matters on gfx1151 — first-run kernel compilation is slow, and
  without this it repeats every session. `run.sh` creates the dir; for compose,
  `mkdir -p` it yourself first so Docker doesn't create it root-owned.
- **Hugging Face models use the host's standard cache**: `~/.cache/huggingface`
  (override with `ROCM_HF_CACHE`) is mounted on top at
  `$HOME/.cache/huggingface`, so models download once per machine and are
  shared with native tools and every other project's containers.
- **Work on a real project**: mount it at `/workspace` with
  `WORKSPACE_DIR=~/projects/my-model ./run.sh pytorch shell` (or set it in
  `.env` for compose) instead of copying files into `workspace/`.
- **Ports** (Jupyter, TensorBoard, …): `ROCM_PORTS="8888:8888" ./run.sh pytorch
  shell` for the wrapper; for compose, uncomment `ports:` in `compose.yaml` and
  use `docker compose run --service-ports <service>`.

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

**Training slow?** Two levers, in order:

1. **bf16** — fp32 attention on gfx1151 is a trap (memory-bandwidth-bound).
2. **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`** — unlocks AOTriton
   mem-efficient SDPA (~11x faster bf16 attention than the math fallback, and
   it stops materializing S×S attention, which is what OOMs training jobs).
   `run.sh` and `compose.yaml` now set it by default; export `=0` to opt out.
   Verified on torch 2.10 / ROCm 7.2.4 — an earlier finding that this flag was
   *slower* predates that stack and is corrected in the findings doc.

With both, gradient checkpointing is usually unnecessary (it was only ever
compensating for math-backend memory). The smoke test times attention under
each backend so a regression is visible. Measurements and history:
[docs/gfx1151-attention-findings.md](docs/gfx1151-attention-findings.md).
Bonus for GEMM-heavy jobs: PyTorch **TunableOp** (`PYTORCH_TUNABLEOP_ENABLED=1`)
finds better GEMM kernels than the untuned gfx1151 defaults (~1.26x measured on
a BERT fine-tune) — tune once per shape-set, then replay the CSV; details in
the findings doc §6.

Recent ROCm supports gfx1151 natively, but some libraries/kernels are only
fully tuned for nearby archs and can throw `invalid device function` or
missing-kernel errors. If that happens, present the GPU as `gfx1100` (RDNA3
desktop) by setting:

```
HSA_OVERRIDE_GFX_VERSION=11.0.0
```

No file edits needed to try it: `run.sh` forwards the variable
(`HSA_OVERRIDE_GFX_VERSION=11.0.0 ./run.sh pytorch check`), and `compose.yaml`
has a commented `environment:` line for it; to bake it into an image there are
commented `ENV` lines in both Dockerfiles. **Try without it first** — the
override can mask real problems and cost performance. Only enable it if the
native path genuinely fails.

**If the image's bundled framework doesn't see gfx1151 at all**, reinstall from
AMD's gfx1151-aware wheel index — the stock PyPI / pytorch.org wheels don't
include gfx1151 kernels.

For **PyTorch**, add to `Dockerfile` after the `FROM` (match the index URL to
your ROCm version):

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
scripts/check.sh     # hardware-free static checks (also run in CI)
.github/workflows/   # CI: runs scripts/check.sh on push / PR
workspace/           # bind-mounted into /workspace (git-ignored)
```

## Adding Python packages

Put extra deps in `requirements.txt` (PyTorch) or `requirements-jax.txt` (JAX)
and rebuild. **Don't** re-add the framework itself — a bare `torch` pulls a
CUDA/CPU wheel, and adding `jax`/`jaxlib`/`jax-rocm7-*` risks clobbering the
ROCm-matched build already in the base image.

## Using this from other projects

The bases are huge (17–23 GB), but Docker stores layers once and shares them —
you pay for a base a single time per machine, and everything built on top costs
only its delta. Treat `framework-rocm:pytorch` / `framework-rocm:jax` as the
machine-wide GPU runtime and use one of two patterns:

**Mount your project in — no image build at all.** Usually all you need:

```bash
WORKSPACE_DIR=~/projects/my-model ./run.sh pytorch shell
pip install --user -r requirements.txt   # inside the container
```

Because `HOME=/workspace`, `pip install --user` lands in your project's
`.local/` on the host — each project keeps its own packages, persisting across
container runs, no image build, no duplicated gigabytes. Model downloads still
go to the shared cache. (Add `.local/` to the project's `.gitignore`.)

**Derive a thin image — when deps should be baked in.** In the project:

```dockerfile
FROM framework-rocm:pytorch
RUN pip install --no-cache-dir -r requirements.txt
```

The derived image *reports* the full ~23 GB, but that's cumulative virtual
size — its unique disk cost is just the installed packages (megabytes).
`docker system df -v` shows the real shared/unique split.

What actually duplicates storage: different **base tags** (a `rocm6.4.4_*`
project and a `rocm7.2.4_*` project share nothing — stay on one tag family),
the **PyTorch and JAX bases** themselves (built separately by AMD, ~40 GB
combined, paid once), and **old bases after a version bump** until you
`docker image prune`.

**uv-managed projects: take torch from the base, not from the lockfile.** A
`uv.lock` pins CPU (or CUDA) torch, and forcing ROCm torch through uv's
source overrides fights the base image's bundled ROCm userspace (learned the
hard way — `libhsa-runtime` mismatches). Inside the container, skip
`uv sync`; install only the non-torch deps with pip and let the base image's
torch/jax stand. For work repos that must stay self-contained, pin the same
upstream `rocm/pytorch` tag in their own Dockerfile — layer sharing comes from
the common base tag, not from deriving from this repo's images.

## Development

`scripts/check.sh` runs the hardware-free checks — shell/Python syntax, a valid
`docker compose config`, and a guard that the default image tags in the README
still match the authoritative `ARG` defaults in the Dockerfiles. Run it before
pushing:

```bash
./scripts/check.sh
```

CI (`.github/workflows/checks.yml`) runs the same script on every push and PR.
The GPU smoke tests aren't in CI — they need a real gfx1151 machine — so run
`./run.sh {pytorch|jax} check` locally after changing anything that touches the
runtime.

## License

MIT — see [LICENSE](LICENSE).
