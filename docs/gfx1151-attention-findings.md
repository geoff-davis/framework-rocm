# gfx1151 attention + training performance findings (2026-07-02)

Discovered while fine-tuning a `BAAI/bge-base-en-v1.5` (110M-param BERT) sentence
encoder on the Framework Desktop (Strix Halo / Radeon 8060S, **gfx1151**). The
original headline: **attention-heavy training is ~10× slower than a CUDA
flash-attention GPU, and the single biggest lever is bf16 — not the torch/ROCm
version.** **2026-07-05 update: see §6 — AOTriton mem-efficient SDPA
(`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`) now works and supersedes the
§1/§3/§4 conclusions.**

## 1. ~~There is no working flash / mem-efficient attention kernel for gfx1151~~

> **SUPERSEDED by §6 (2026-07-05):** on torch 2.10 / ROCm 7.2.4 the AOTriton
> mem-efficient kernel behind `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is
> ~11x FASTER than the math fallback in bf16. The "makes it worse" result
> below was measured on the older stack / fp32-dominated runs.

Every torch build we tested (2.9.1 and 2.10, ROCm 7.0 / 7.2.0 / 7.2.4) emits:

```
UserWarning: Mem Efficient attention on Current AMD GPU is still experimental.
Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1.
(Triggered internally at .../transformers/hip/sdp_utils.cpp)
```

So `scaled_dot_product_attention` silently falls back to the **math backend**
(a plain O(n²) softmax·matmul), which is memory-bandwidth-bound and slow.

- **`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` makes it WORSE, not better.**
  On a bge-base fine-tune it went from ~70 s/step → ~220 s/step (the AOTriton
  gfx1151 kernels are unusably slow / compile-thrash). **Do not enable it.**

## 2. bf16 is the lever; the torch/ROCm version barely matters

Measured wall-clock per training step (bge-base, batch 64, seq≤512, MNRL loss,
gradient-checkpointing ON — see §3):

| Stack | fp32 | bf16 |
| --- | ---: | ---: |
| Docker `rocm7.2.4` / torch 2.10 | ~70 s/step | **~9.5 s/step** |
| Native venv, torch 2.10 + rocm7.0 wheel (`LD_PRELOAD` rocm-7.2.0 libhsa) | ~28 s/step | **~7.6 s/step** |
| Docker `rocm7.2.0` / torch 2.9.1 | ~30 s/step | (≈ same as native) |

Takeaways:
- **fp32 is a trap on gfx1151** — the math-attention path is especially
  bandwidth-bound in fp32. bf16 roughly quarters it (halves matmul + halves the
  bandwidth), landing everything in the ~8–10 s/step band.
- Nuance: the *isolated* SDPA micro-benchmark in `check_gpu.py` shows fp32 ≈
  bf16 (~125 ms/iter either way at B32·H12·S512·D64) — the math backend
  upcasts internally, so SDPA itself barely benefits. The end-to-end bf16 win
  comes from everything around it (linear layers, reduced bandwidth). Don't
  read smoke-test parity as contradicting the training numbers above.
- The `rocm7.2.4` image's fp32 is anomalously slow (70 s vs 28 s), but **bf16
  converges the stacks** to ~8–10 s/step, so the version choice is a wash for
  speed once you're in bf16. Pick the version for hygiene (see README), not
  perf.
- gfx1151 (RDNA 3.5) supports bf16 natively — no accuracy surprises; bf16
  training of a small BERT is numerically fine.

## 3. The GPU only exposes ~61 GiB → ~~gradient-checkpointing is mandatory~~

> **SUPERSEDED by §6 (2026-07-05):** the ~61 GiB carveout is real, but the OOM
> was the math backend materializing S×S attention for two MNRL forward
> graphs. With AOTriton mem-efficient SDPA the same job fits WITHOUT
> gradient-checkpointing (which costs ~1.4x step time).

Despite 128 GB of unified system RAM, the GPU VRAM carveout (BIOS UMA / GTT
split) presents as **~61.4 GiB** to ROCm. A batch-64 bge-base fine-tune **OOMs
without gradient-checkpointing** (`HIP out of memory ... 61.42 GiB total`).
With `gradient_checkpointing=True` it fits comfortably in bf16. If you need
bigger batches or models, raise the UMA carveout in BIOS or keep grad-ckpt on.

## 4. ~~Recommendation for attention-heavy training on this box~~ (superseded by §6)

1. **bf16 + gradient-checkpointing.** Non-negotiable for tractable training.
2. **Do not** set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.
3. Expect ~10× slower attention than a CUDA flash GPU until a real flash /
   mem-efficient kernel ships for gfx1151. For a 110M encoder that's ~9 s/step
   at batch 64 (~14–18 h for a 3-epoch, ~140k-pair run).
4. `check_gpu.py` now includes an **attention timing** section (fp32 vs bf16
   SDPA + backend report) so this class of regression is visible from the smoke
   test — the old matmul-only check sailed right past it.

## 5. Bonus hygiene finding: `compose.yaml` defaulted to root (FIXED)

> **FIXED since:** `compose.yaml` now *requires* `HOST_UID`/`HOST_GID`
> (`${HOST_UID:?...}` — compose refuses to start without them), so the silent
> root default described below can't happen anymore. Kept for the war story.

`compose.yaml` sets `user: "${HOST_UID:-0}:${HOST_GID:-0}"` — a `docker compose
up` **without** `HOST_UID`/`HOST_GID` exported runs as **root** and writes
root-owned files into the **shared** `~/.cache/huggingface` (observed:
root-owned `models--BAAI--bge-large-en-v1.5`). That can later block non-root
runs (of this project or others sharing the HF cache) from
updating those models. `run.sh` is fine (it uses `--user $(id -u):$(id -g)`).
Suggested fix: default the compose UID/GID to the host user, or document that
`HOST_UID`/`HOST_GID` must be exported. Clear stale root files with
`sudo chown -R "$USER" ~/.cache/huggingface`.

## 6. CORRECTION (2026-07-05): enable AOTriton mem-efficient SDPA — it's now the biggest attention lever

Re-measured on the current default stack (`rocm/pytorch:rocm7.2.4…pytorch_release_2.10.0`,
torch 2.10.0, warm kernel caches), while taking a downstream bge-base
fine-tune from ~9.8 → 1.10 s/step (~9x). The §1 "makes it WORSE" finding does
not reproduce there; §1/§3/§4 above are kept for history but superseded.

### Micro-benchmark (`check_gpu.py`, SDPA fwd+bwd, B32·H12·S512·D64)

| | flag unset (math) | `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` |
| --- | ---: | ---: |
| fp32 | 117.2 ms/iter | 69.7 ms/iter (~1.7x) |
| bf16 | 92.3 ms/iter | **8.4 ms/iter (~11x)** |

The smoke test now pins each backend explicitly (`bf16 math-only` vs
`bf16 mem-effic.`), so kernel availability and the gap are printed directly
instead of inferred from warnings.

### End-to-end (110M BERT sentence-encoder fine-tune, MNRL, batch 64, bf16)

- math backend, seq≤512, grad-ckpt ON (the §4 recipe): **9.8 s/step**
- drop grad-ckpt with math backend: **HIP OOM at ~61 GiB** — the real culprit
  behind §3: math SDPA materializes S×S attention per layer, and contrastive
  losses hold two forward graphs at once.
- AOTriton + no grad-ckpt + seq cap 256 + 4 dataloader workers: **1.39 s/step**
- plus PyTorch TunableOp (tune once per GEMM-shape-set, replay CSV; pad batches
  to a multiple of 64 to bound the shape count): **1.10 s/step (~9x total)**
- Numerics: 20-step same-seed loss A/B old vs new config matched within ~0.4%.

### Why the 2026-07-02 measurement said the opposite

Best explanation: it was observed on fp32-heavy runs / the older
torch 2.9.1 + ROCm 7.2.0 images, and likely paid first-run AOTriton kernel
autotune ("compile-thrash") inside a short probe. On torch 2.10/ROCm 7.2.4
with warm caches the kernels are simply good. If you see the old pathology on
some other stack, report the exact base tag before concluding anything.

### Standing recommendations (replace §4)

1. **bf16 + `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`** — `run.sh` and
   `compose.yaml` now default the flag on (export `=0` to opt out).
2. **Don't reach for gradient checkpointing by default** — with mem-efficient
   attention it's usually unneeded and costs ~1.4x step time. Keep it for
   genuinely bigger-than-VRAM jobs.
3. **TunableOp for GEMM-heavy jobs**: `PYTORCH_TUNABLEOP_ENABLED=1` +
   `PYTORCH_TUNABLEOP_TUNING=1` once (writes a per-arch CSV; TunableOp appends
   the device ordinal to the filename), then replay with `TUNING=0`. ~1.26x on
   the BERT fine-tune. Keep GEMM shapes bounded (fixed batch, pad seq widths
   to a multiple) or tuning never converges.
4. torch.compile works once `TRITON_CACHE_DIR` points somewhere writable (the
   wrappers now set it into the persistent cache mount) but measured below
   TunableOp on this workload (~7% over AOTriton alone) — benchmark before
   adopting.

## 7. JAX (2026-07-05): Pallas flash attention works on gfx1151 — smaller win, not automatic

Same shape as §6 (B32·H12·S512·D64, bf16, fwd+bwd, jax 0.8.2 /
`rocm/jax:rocm7.2.4`):

| path | ms/iter |
| --- | ---: |
| `jax.nn.dot_product_attention` (XLA math; `implementation='cudnn'` is rejected on this GPU) | 30.7 |
| Pallas flash attention (`jax.experimental.pallas.ops.gpu.attention.mha`, Triton-backed) | **16.6** |

Context against §6's torch numbers: XLA's "math" baseline (30.7 ms) is already
~3x faster than torch's math backend (92.3 ms) — XLA fuses the softmax·matmul
chain — so the Triton lever buys JAX only ~1.9x where torch gained ~11x. The
AMD-tuned AOTriton kernels also still beat generic Pallas codegen ~2x
(8.4 vs 16.6 ms).

Practical differences from the torch flag:

- **Not automatic.** There is no env var; `jax.nn.dot_product_attention` will
  not use it. You must call the Pallas `mha` op (or use a library that wires
  it up) explicitly.
- These are timing probes — validate numerics against the XLA path before
  adopting (`jnp.allclose` on fwd + grads), and expect block-size tuning to
  matter at other shapes.
