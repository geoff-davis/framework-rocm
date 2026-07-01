#!/usr/bin/env bash
# Convenience wrapper for the plain `docker` path (no compose needed).
#
#   ./run.sh build            # build the image
#   ./run.sh shell            # interactive shell with the GPU attached
#   ./run.sh check            # run the GPU smoke test
#   ./run.sh <any command>    # run an arbitrary command in the container
#
# The docker run flags here are the Framework Desktop / ROCm essentials:
# device passthrough for /dev/kfd and /dev/dri, membership in the render/video
# groups, and a relaxed seccomp profile that ROCm's userspace requires.
set -euo pipefail

IMAGE="framework-rocm:local"
ROCM_PYTORCH_TAG="${ROCM_PYTORCH_TAG:-rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.6.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

case "${1:-shell}" in
  build)
    docker build --build-arg "ROCM_PYTORCH_TAG=${ROCM_PYTORCH_TAG}" -t "${IMAGE}" "${HERE}"
    ;;
  shell)
    docker_run /bin/bash
    ;;
  check)
    docker_run python /usr/local/bin/check_gpu.py
    ;;
  *)
    docker_run "$@"
    ;;
esac
