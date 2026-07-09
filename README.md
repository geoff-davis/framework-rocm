# framework-rocm

Reproducible ROCm containers for the **Framework Desktop** — the AMD Ryzen AI
Max ("Strix Halo") board with the integrated **Radeon 8060S** GPU (`gfx1151`).
It exists because getting a recent ROCm and a matching PyTorch or JAX talking to
this GPU is fiddly, and re-deriving the right versions every time is a waste of
an afternoon.

Two sibling images, each built on the matching AMD-published base image, so
ROCm and the framework arrive pre-pinned and mutually consistent (and verified
on this hardware — see below):

- **`pytorch`** — from [`rocm/pytorch`](https://hub.docker.com/r/rocm/pytorch)
- **`jax`** — from [`rocm/jax`](https://hub.docker.com/r/rocm/jax)

They're kept separate on purpose: each framework bundles its own ROCm userspace
libs, so separate images avoid version friction and keep pulls smaller. Both
share the same device passthrough, gfx1151 specifics, and validation pattern.

**Measured on this hardware** (see
[docs/gfx1151-attention-findings.md](docs/gfx1151-attention-findings.md) for
the full story): enabling AOTriton mem-efficient SDPA — which these containers
now do by default — takes bf16 attention from 92 → 8.4 ms/iter (~11x) at
BERT-base shape, and a real 110M-param sentence-encoder fine-tune from
9.8 → 1.10 s/step (~9x) once combined with the other levers documented there
(no gradient checkpointing, TunableOp, seq-length cap).

> **gfx1151 support, honestly:** AMD's
> [Radeon/Ryzen Linux matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)
> lists the Ryzen AI Max+ 395 / Radeon 8060S (gfx1151) with production support
> on **ROCm 7.2.1 + PyTorch 2.9.1** (FP16 is what's officially validated).
> This repo defaults to AMD's *newer published images* (ROCm 7.2.4 base;
> PyTorch 2.10.0 / JAX 0.8.2) — a combination AMD ships but does not list in
> that matrix — because it measures faster and cleaner here (see the findings
> doc). If you want to stay strictly on the AMD-validated combo, pin a
> `rocm7.2.1`/torch-2.9.1 tag instead (see below). Cross-check the
> [general ROCm matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html)
> when bumping tags — framework support lags the ROCm release.
>
> Both images here were **verified on an actual Framework Desktop**
> (last verified **2026-07-05**, ROCm 7.2.4 / torch 2.10.0 / jax 0.8.2): the
> `rocm/pytorch` and `rocm/jax` bases both see gfx1151 (Radeon 8060S) and
> run GPU compute *natively* — no `HSA_OVERRIDE_GFX_VERSION` and no gfx1151
> fallback wheels required.

## Requirements (host)

- A Framework Desktop (Ryzen AI Max / Strix Halo) — or any gfx1151 machine.
- A recent kernel with `amdgpu` loaded (`/dev/kfd` and `/dev/dri/renderD*` present).
- Docker with the Compose plugin.

The wrapper and Compose add the device nodes' numeric owner groups inside the
container, so host group *names* and pre-existing `render`/`video` membership
are not assumed. Your user still needs permission to run Docker itself.

**Preflight** — each of these should succeed *before* you build anything (they
catch most first-run failures without waiting on a multi-GB pull):

```bash
docker version && docker compose version      # docker + compose installed?
docker run --rm hello-world                   # can run containers (no sudo)?
ls -l /dev/kfd /dev/dri/renderD*              # GPU device nodes exist?
stat -c '%g %n' /dev/kfd /dev/dri/renderD*    # numeric GIDs compose needs
```

## Quick start

Pick a framework — `pytorch` or `jax`.

> **Before the first build:** the AMD base images are **17–23 GB**. The first
> build/pull downloads that once — expect tens of minutes on a typical
> connection — and Docker then stores the layers a single time and shares them
> machine-wide (`docker system df` shows usage). Make sure the disk has room.

The easiest path is the `run.sh` wrapper — no `.env` to edit, no Compose
knowledge needed; it resolves the device GIDs and maps your host user
automatically (first arg is the framework):

```bash
./run.sh pytorch build
./run.sh pytorch check          # quick deterministic GPU correctness check
./run.sh jax     check          # quick deterministic GPU correctness check
./run.sh pytorch bench          # correctness check + attention benchmark
./run.sh jax     bench          # same for JAX (includes XLA compilation)
./run.sh pytorch shell          # interactive shell
./run.sh jax     python your_script.py
```

The first `bench` after an image or shape change can spend a minute compiling
GPU kernels; subsequent runs reuse the persistent cache described below. On
PyTorch it also fails if bf16 efficient SDPA exceeds 25 ms at the documented
shape (verified at ~8.4 ms); set `MAX_BF16_EFFICIENT_MS=` to disable that guard
or choose a different positive limit for other hardware.

**Already a Compose user?** The same containers are defined in `compose.yaml`.
This path needs a `.env` with host-specific GIDs + your UID/GID (`run.sh` does
not):

```bash
cp .env.example .env      # then edit, or just generate it:
mkdir -p ~/.cache/framework-rocm ~/.cache/huggingface && \
printf 'RENDER_GID=%s\nVIDEO_GID=%s\nHOST_UID=%s\nHOST_GID=%s\n' \
  "$(stat -c %g /dev/dri/renderD* | head -1)" "$(stat -c %g /dev/kfd)" \
  "$(id -u)" "$(id -g)" > .env
```

Then:

```bash
docker compose build pytorch                                        # or: jax
docker compose run --rm pytorch python /usr/local/bin/check_gpu.py  # quick check
docker compose run --rm jax     python /usr/local/bin/check_jax.py  # quick check
docker compose run --rm pytorch python /usr/local/bin/check_gpu.py --bench
docker compose run --rm jax     python /usr/local/bin/check_jax.py --bench
docker compose run --rm pytorch                                     # interactive shell
```

<details>
<summary><strong>New to Docker? 60-second glossary</strong></summary>

- **Image** — the frozen filesystem snapshot you build once (`framework-rocm:pytorch`).
- **Container** — a running (disposable) instance of an image.
- **`--rm`** — deletes the *container* when it exits. The *image* stays; so do
  files written into bind-mounted paths.
- **Bind mount** (`-v host:container`) — a host directory appearing inside the
  container. Writes there land on the host and persist; writes anywhere else
  vanish with the container.
- **Build / tag** — `docker build` produces an image; the tag
  (`name:variant`) is just its label. Rebuilding with the same tag replaces it.
- **Compose** — a YAML wrapper (`compose.yaml`) that stores the run flags so
  you don't retype them; `docker compose run pytorch` ≈ the long `docker run`
  in `run.sh`.

</details>

A successful PyTorch correctness check looks roughly like:

```
torch version : 2.10.0+rocm7.2.4.git3d3aa833
ROCm/HIP ver  : 7.2.53211
device count  : 1
  [0] Radeon 8060S Graphics
gpu arch      : gfx1151
matmul OK     : sum=1073741824.000 on Radeon 8060S Graphics
```

…and the JAX one:

```
jax version   : 0.8.2
jaxlib version: 0.8.2+rocm7.2.4
devices       : [RocmDevice(id=0)]
device kind   : Radeon 8060S Graphics
matmul OK     : sum=1073741824.0 on rocm:0
```

Both checks validate the deterministic matmul result and verify that the device
matches expectations, so corrupt compute cannot pass merely because a kernel
launched. PyTorch checks the arch string
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

### Where are my files?

Concretely, with the defaults (`HOME=/workspace` inside the container):

| You write… | …it lands on the host at | Survives exit? |
| --- | --- | --- |
| `/workspace/foo.py` | `./workspace/foo.py` (or `$WORKSPACE_DIR`) | yes |
| `pip install --user …` | `./workspace/.local/` | yes |
| model downloads (`$HOME/.cache/huggingface`) | `~/.cache/huggingface` | yes |
| other `$HOME/.cache` (pip, MIOpen, Triton) | `~/.cache/framework-rocm` | yes |
| anything else — `apt install`, `/tmp`, system site-packages | nowhere | **no** — gone when the `--rm` container exits; bake it into the image instead |

## What makes the GPU visible

These aren't optional decorations — they're why compute works inside the
container. The flip side: `seccomp=unconfined`, `ipc=host`, direct GPU device
access, and host cache/workspace mounts mean these containers are **not a
security boundary** — treat code you run in them like code you'd run on the
host, and don't point them at untrusted models/notebooks you wouldn't run
natively:

| Flag | Why |
| --- | --- |
| `--device=/dev/kfd` | ROCm's kernel fusion driver (compute). |
| `--device=/dev/dri` | GPU render nodes. |
| `--group-add render` / `video` | Own `/dev/kfd` and `/dev/dri/render*`. |
| `--security-opt seccomp=unconfined` | ROCm userspace trips the default seccomp profile. |
| `--ipc=host` | Avoids shared-memory limits for larger tensors / dataloaders. |

`run.sh` reads the numeric owners directly from `/dev/kfd` and
`/dev/dri/renderD*`. For Compose, put those numeric GIDs in `.env` using the
generator in Quick start; group names are not assumed.

## Troubleshooting

| Symptom | Likely cause → fix |
| --- | --- |
| `permission denied … /var/run/docker.sock` | Your user isn't in the `docker` group → `sudo usermod -aG docker "$USER"`, re-login. |
| `ls: cannot access '/dev/kfd'` | `amdgpu` not loaded / kernel too old → `lsmod \| grep amdgpu`, check `dmesg`, update kernel/firmware. |
| Compose: `set VIDEO_GID in .env` | No `.env` → generate it (one-liner in Quick start / `.env.example`). |
| `torch.cuda.is_available()` is `False` in the container | Ran without the device/group flags → use `run.sh` or Compose, not bare `docker run`; verify numeric ownership with `stat -c '%g %n' /dev/kfd /dev/dri/renderD*`. |
| Files under `~/.cache/huggingface` owned by root | A root-run container wrote them → `sudo chown -R "$USER" ~/.cache/huggingface`; keep `HOST_UID`/`HOST_GID` set (compose refuses to default to root). |
| Jupyter/TensorBoard unreachable | Port not published → `ROCM_PORTS="8888:8888" ./run.sh …` or `docker compose run --service-ports …`. |
| Disk full after pulls | See usage with `docker system df`; reclaim with `docker image prune` (dangling only) — `docker system prune -a` also deletes the 17–23 GB bases you'd re-download. |
| Attention/training unexpectedly slow | Run `./run.sh pytorch bench`, then see "The gfx1151 gotcha" below (bf16 + AOTriton). |

## Picking versions

Each image's base reference is a build arg. Browse tags, build against one, and
rerun the correctness check and benchmark.

**PyTorch** — `ROCM_PYTORCH_TAG`, default immutable reference
`rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e`
([tags](https://hub.docker.com/r/rocm/pytorch/tags)):

```bash
docker compose build --build-arg ROCM_PYTORCH_TAG=<tag> pytorch
# or:  ROCM_PYTORCH_TAG=<tag> ./run.sh pytorch build
```

- Default is newest / best real-world Strix Halo behaviour.
- Older PyTorch on the same ROCm: swap `_2.10.0` for `_2.9.1` / `_2.8.0` / `_2.7.1`.
- AMD-validated combo (per the Ryzen matrix — see the support note up top):
  `rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`.

**JAX** — `ROCM_JAX_TAG`, default immutable reference
`rocm7.2.4-jax0.8.2-py3.12@sha256:6a16c6afc317745f2f04519e78ca292eacd8e768c031e1fc856c9025907f6a1f`
([tags](https://hub.docker.com/r/rocm/jax/tags)):

```bash
docker compose build --build-arg ROCM_JAX_TAG=<tag> jax
# or:  ROCM_JAX_TAG=<tag> ./run.sh jax build
```

Keep the JAX / jaxlib / `jax-rocm7-plugin` versions aligned with the ROCm major
— that mismatch is the usual JAX-on-ROCm failure mode. AMD's `rocm/jax` tags
already bundle a matched set.

The defaults include registry digests so the same source always selects the
same AMD image. A plain tag override is convenient for testing but mutable;
once verified, use `<tag>@sha256:<digest>` (obtain it with `docker buildx
imagetools inspect <image>:<tag>`) for a repeatable build.

## The gfx1151 gotcha

**Training slow?** Two levers, in order:

1. **bf16** — fp32 attention on gfx1151 is a trap (memory-bandwidth-bound).
2. **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`** — unlocks AOTriton
   mem-efficient SDPA (~11x faster bf16 attention than the math fallback, and
   it stops materializing S×S attention, which is what OOMs training jobs).
   `run.sh` and `compose.yaml` set it by default; export `=0` to opt out.
   Verified on torch 2.10 / ROCm 7.2.4 — an earlier finding that this flag was
   *slower* predates that stack and is corrected in the findings doc.

With both, gradient checkpointing is usually unnecessary (it was only ever
compensating for math-backend memory). The explicit `bench` action times
attention under each backend and checks numerical agreement so a regression is
visible. Measurements and history:
[docs/gfx1151-attention-findings.md](docs/gfx1151-attention-findings.md).
Bonus for GEMM-heavy jobs: PyTorch **TunableOp** (`PYTORCH_TUNABLEOP_ENABLED=1`)
finds better GEMM kernels than the untuned gfx1151 defaults (~1.26x measured on
a BERT fine-tune) — tune once per shape-set, then replay the CSV; details in
the findings doc §6. The wrapper and Compose forward the TunableOp variables,
so a host-side prefix works, for example:

```bash
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 \
  ./run.sh pytorch python train.py
```

Recent ROCm supports gfx1151 natively, but some libraries/kernels are only
fully tuned for nearby archs and can throw `invalid device function` or
missing-kernel errors. If that happens, present the GPU as `gfx1100` (RDNA3
desktop) by setting:

```
HSA_OVERRIDE_GFX_VERSION=11.0.0
```

No file edits needed to try it: `run.sh` forwards the variable
(`HSA_OVERRIDE_GFX_VERSION=11.0.0 ./run.sh pytorch check`), and `compose.yaml`
forwards it when set in the shell or `.env`; to bake it into an image there are
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

Run the correctness check and benchmark afterward. If the base image already
passes, you don't need this.

## Layout

```
Dockerfile           # PyTorch image: FROM rocm/pytorch + gfx1151 env
Dockerfile.jax       # JAX image:     FROM rocm/jax     + gfx1151 env
compose.yaml         # two services sharing GPU passthrough (YAML anchor)
run.sh               # plain-docker path: ./run.sh {pytorch|jax} {build|shell|check|…}
requirements.txt     # extra deps for the PyTorch image (keep minimal)
requirements-jax.txt # extra deps for the JAX image (keep minimal)
requirements-dev.txt # exact Ruff/ShellCheck versions for local checks and CI
requirements-pytorch.lock # generated target-wheel versions + SHA-256 hashes
requirements-rocm-base.txt # framework packages owned only by the AMD base
ruff.toml             # Python lint and formatting policy
check_gpu.py         # PyTorch deterministic GPU check + optional SDPA benchmark
check_jax.py         # JAX deterministic GPU check + optional attention benchmark
constraints-pytorch.txt # verified versions feeding lock generation
scripts/lock_dependencies.py # target-specific lock generator and verifier
tests/               # hardware-free regression tests for helpers/configuration
scripts/check.sh     # hardware-free static checks (also run in CI)
.github/workflows/   # CI: runs scripts/check.sh on push / PR
workspace/           # bind-mounted into /workspace (git-ignored)
```

## Adding Python packages

Put direct dependencies in `requirements.txt` (PyTorch) or
`requirements-jax.txt` (JAX). For PyTorch, update `constraints-pytorch.txt`
with the deliberately selected versions, then regenerate the binary-only,
target-specific hash lock:

```bash
uv run --isolated --python 3.12 --with-requirements requirements-dev.txt \
  python scripts/lock_dependencies.py
```

The generator runs only on CPython 3.12/Linux x86_64, resolves the complete
non-ROCm closure, selects the exact wheel using the container's glibc 2.39
compatibility ceiling rather than the host's potentially newer libc, and
records its SHA-256 hash in `requirements-pytorch.lock`. Docker installs that
lock with `--require-hashes --only-binary=:all: --no-deps`, then runs `pip
check` and verifies that torch still reports a HIP runtime. `--no-deps` is safe
here because every non-ROCm transitive dependency is an explicit lock entry;
it also prevents pip from resolving a PyPI torch build.

Review the generated lock diff, rebuild, and run `./run.sh pytorch bench`.
**Don't** re-add the framework itself — a bare `torch` pulls a
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

`scripts/check.sh` runs the hardware-free checks — ShellCheck, Ruff lint and
format validation, shell/Python syntax, unit tests for result validation and
wrapper/lock configuration, a valid `docker compose config`, environment
forwarding, target-wheel hash and ROCm-exclusion checks, and a guard that the
default image references in the README match the authoritative Dockerfile
defaults. Run it with the exact tool versions from
`requirements-dev.txt` before pushing (the isolated environment leaves the
ROCm/runtime Python installation alone):

```bash
uv run --isolated --with-requirements requirements-dev.txt ./scripts/check.sh
```

If you already installed `requirements-dev.txt` into an active development
environment, running `./scripts/check.sh` directly is equivalent.

CI (`.github/workflows/checks.yml`) runs the same script on every push and PR.
Third-party actions are pinned to immutable Git commit SHAs; keep the readable
version comments and SHAs together when updating them. The static checks reject
mutable action tags.
The GPU checks aren't in CI — they need a real gfx1151 machine — so run
`./run.sh {pytorch|jax} check` locally after runtime changes, and run the
corresponding `bench` action after framework, kernel, or performance changes.

## License

MIT — see [LICENSE](LICENSE).
