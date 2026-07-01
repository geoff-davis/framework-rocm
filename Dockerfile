# Framework Desktop (Strix Halo / Radeon 8060S, gfx1151) ROCm + PyTorch image.
#
# We build on top of AMD's official rocm/pytorch image so ROCm, PyTorch, and
# their supporting libraries are already pinned to a known-good combination.
# Bump ROCM_PYTORCH_TAG to move to a newer ROCm/PyTorch; see README for how to
# find a tag that supports gfx1151.
ARG ROCM_PYTORCH_TAG=rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
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
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# A quick GPU visibility check available on the PATH inside the container.
COPY check_gpu.py /usr/local/bin/check_gpu.py
RUN chmod +x /usr/local/bin/check_gpu.py

CMD ["/bin/bash"]
