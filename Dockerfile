# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89
# Framework Desktop (Strix Halo / Radeon 8060S, gfx1151) ROCm + PyTorch image.
#
# We build on top of AMD's official rocm/pytorch image so ROCm, PyTorch, and
# their supporting libraries are already pinned to a known-good combination.
# Bump ROCM_PYTORCH_TAG to move to a newer ROCm/PyTorch; see README for how to
# find a tag that supports gfx1151.
# The default tag is paired with the immutable digest verified on this machine.
# An override may still be a plain tag for experimentation, or tag@sha256:...
# when the resulting build also needs to be reproducible.
ARG ROCM_PYTORCH_TAG=rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:4449f856653602317e4101a76fce599c7fcd58ccec2e539951fce5f73083179e
FROM rocm/pytorch:${ROCM_PYTORCH_TAG}

# gfx1151 (Strix Halo) is supported by recent ROCm, but some kernels/libraries
# still fall back to the nearest fully-tuned arch. If you hit "invalid device
# function" or missing-kernel errors, uncomment the override below to present
# the GPU as gfx1100 (RDNA3 desktop). Leave it off first and only enable if
# needed — the override can hurt performance or mask real issues.
# ENV HSA_OVERRIDE_GFX_VERSION=11.0.0

# Make the target arch explicit for anything that compiles kernels at runtime.
ENV PYTORCH_ROCM_ARCH=gfx1151

WORKDIR /workspace

# Extra Python deps go here so they're baked into the image and cached.
COPY requirements.txt /tmp/requirements.txt
COPY constraints-pytorch.txt /tmp/constraints-pytorch.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt -c /tmp/constraints-pytorch.txt && \
    pip check

# A deterministic GPU correctness check available on the PATH in the container.
COPY check_gpu.py /usr/local/bin/check_gpu.py
RUN chmod +x /usr/local/bin/check_gpu.py

CMD ["/bin/bash"]
