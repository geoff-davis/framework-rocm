#!/usr/bin/env bash
# Convenience wrapper for the plain `docker` path (no compose needed).
#
#   ./run.sh pytorch build         # build the PyTorch image
#   ./run.sh jax     build         # build the JAX image
#   ./run.sh pytorch check         # run the PyTorch GPU smoke test
#   ./run.sh jax     check         # run the JAX GPU smoke test
#   ./run.sh pytorch shell         # interactive shell with the GPU attached
#   ./run.sh jax     python x.py   # run an arbitrary command in the JAX image
#
# The docker run flags here are the Framework Desktop / ROCm essentials:
# device passthrough for /dev/kfd and /dev/dri, membership in the render/video
# groups, and a relaxed seccomp profile that ROCm's userspace requires.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FRAMEWORK="${1:-pytorch}"
shift || true
ACTION="${1:-shell}"
shift || true

case "${FRAMEWORK}" in
  pytorch)
    IMAGE="framework-rocm:pytorch"
    DOCKERFILE="Dockerfile"
    BUILD_ARG="ROCM_PYTORCH_TAG=${ROCM_PYTORCH_TAG:-rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0}"
    CHECK="/usr/local/bin/check_gpu.py"
    ;;
  jax)
    IMAGE="framework-rocm:jax"
    DOCKERFILE="Dockerfile.jax"
    BUILD_ARG="ROCM_JAX_TAG=${ROCM_JAX_TAG:-rocm7.2.4-jax0.8.2-py3.12}"
    CHECK="/usr/local/bin/check_jax.py"
    ;;
  *)
    echo "usage: $0 {pytorch|jax} {build|shell|check|<command...>}" >&2
    exit 2
    ;;
esac

docker_run() {
  docker run --rm -it \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --security-opt seccomp=unconfined \
    --ipc=host \
    -v "${HERE}/workspace:/workspace" \
    "${IMAGE}" "$@"
}

case "${ACTION}" in
  build)
    docker build -f "${HERE}/${DOCKERFILE}" --build-arg "${BUILD_ARG}" -t "${IMAGE}" "${HERE}"
    ;;
  shell)
    docker_run /bin/bash
    ;;
  check)
    docker_run python "${CHECK}"
    ;;
  *)
    docker_run "${ACTION}" "$@"
    ;;
esac
