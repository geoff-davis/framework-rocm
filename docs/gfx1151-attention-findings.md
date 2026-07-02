# gfx1151 attention + training performance findings (2026-07-02)

Discovered while fine-tuning a `BAAI/bge-base-en-v1.5` (110M-param BERT) sentence
encoder on the Framework Desktop (Strix Halo / Radeon 8060S, **gfx1151**). The
headline: **attention-heavy training is ~10× slower than a CUDA flash-attention
GPU, and the single biggest lever is bf16 — not the torch/ROCm version.**

## 1. There is no working flash / mem-efficient attention kernel for gfx1151

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

## 3. The GPU only exposes ~61 GiB → gradient-checkpointing is mandatory

Despite 128 GB of unified system RAM, the GPU VRAM carveout (BIOS UMA / GTT
split) presents as **~61.4 GiB** to ROCm. A batch-64 bge-base fine-tune **OOMs
without gradient-checkpointing** (`HIP out of memory ... 61.42 GiB total`).
With `gradient_checkpointing=True` it fits comfortably in bf16. If you need
bigger batches or models, raise the UMA carveout in BIOS or keep grad-ckpt on.

## 4. Recommendation for attention-heavy training on this box

1. **bf16 + gradient-checkpointing.** Non-negotiable for tractable training.
2. **Do not** set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.
3. Expect ~10× slower attention than a CUDA flash GPU until a real flash /
   mem-efficient kernel ships for gfx1151. For a 110M encoder that's ~9 s/step
   at batch 64 (~14–18 h for a 3-epoch, ~140k-pair run).
4. `check_gpu.py` now includes an **attention timing** section (fp32 vs bf16
   SDPA + backend report) so this class of regression is visible from the smoke
   test — the old matmul-only check sailed right past it.

## 5. Bonus hygiene finding: `compose.yaml` defaults to root

`compose.yaml` sets `user: "${HOST_UID:-0}:${HOST_GID:-0}"` — a `docker compose
up` **without** `HOST_UID`/`HOST_GID` exported runs as **root** and writes
root-owned files into the **shared** `~/.cache/huggingface` (observed:
root-owned `models--BAAI--bge-large-en-v1.5`). That can later block non-root
runs (of this project or others sharing the HF cache, e.g. other projects) from
updating those models. `run.sh` is fine (it uses `--user $(id -u):$(id -g)`).
Suggested fix: default the compose UID/GID to the host user, or document that
`HOST_UID`/`HOST_GID` must be exported. Clear stale root files with
`sudo chown -R "$USER" ~/.cache/huggingface`.
