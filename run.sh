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
#
# Env overrides:
#   ROCM_PYTORCH_TAG / ROCM_JAX_TAG  build a non-default base image tag
#                                    (default lives in the Dockerfile ARG)
#   ROCM_ROOT=1                      run as root instead of your host user
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
    TAG_ENV="ROCM_PYTORCH_TAG"   # canonical default lives in the Dockerfile ARG
    CHECK="/usr/local/bin/check_gpu.py"
    ;;
  jax)
    IMAGE="framework-rocm:jax"
    DOCKERFILE="Dockerfile.jax"
    TAG_ENV="ROCM_JAX_TAG"       # canonical default lives in the Dockerfile ARG
    CHECK="/usr/local/bin/check_jax.py"
    ;;
  *)
    echo "usage: $0 {pytorch|jax} {build|shell|check|<command...>}" >&2
    exit 2
    ;;
esac

# Resolve the host's render/video GIDs numerically. The container's group file
# usually has no `render`/`video` entry, so passing the names to --group-add
# fails ("unable to find group render"); the numeric GID always works. Falls
# back to the name if the group isn't found on the host.
host_gid() {
  getent group "$1" | cut -d: -f3 | grep . || echo "$1"
}
RENDER_GID="$(host_gid render)"
VIDEO_GID="$(host_gid video)"

docker_run() {
  # Only request an interactive TTY when we actually have one, so `check` and
  # scripted commands work from non-interactive shells (CI, background jobs).
  local tty_flags=()
  if [ -t 0 ] && [ -t 1 ]; then
    tty_flags=(-it)
  fi
  # Map the container process to the host user by default so files written to
  # the bind-mounted workspace aren't root-owned. ROCM_ROOT=1 keeps root (needed
  # if you pip-install into system site-packages inside the container).
  local user_flags=()
  if [ -z "${ROCM_ROOT:-}" ]; then
    user_flags=(--user "$(id -u):$(id -g)")
  fi
  docker run --rm "${tty_flags[@]}" "${user_flags[@]}" \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add "${VIDEO_GID}" \
    --group-add "${RENDER_GID}" \
    --security-opt seccomp=unconfined \
    --ipc=host \
    -e EXPECTED_ARCH -e STRICT_ARCH -e EXPECTED_DEVICE -e STRICT_DEVICE \
    -v "${HERE}/workspace:/workspace" \
    "${IMAGE}" "$@"
}

case "${ACTION}" in
  build)
    # Pass --build-arg only when the tag is overridden; otherwise use the
    # Dockerfile ARG default (the single source of truth for the base tag).
    build_args=()
    tag_val="${!TAG_ENV:-}"
    if [ -n "${tag_val}" ]; then
      build_args=(--build-arg "${TAG_ENV}=${tag_val}")
    fi
    docker build -f "${HERE}/${DOCKERFILE}" "${build_args[@]}" -t "${IMAGE}" "${HERE}"
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
