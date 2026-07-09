#!/usr/bin/env bash
# Convenience wrapper for the plain `docker` path (no compose needed).
#
#   ./run.sh pytorch build         # build the PyTorch image
#   ./run.sh jax     build         # build the JAX image
#   ./run.sh pytorch check         # run the quick PyTorch GPU correctness check
#   ./run.sh jax     check         # run the quick JAX GPU correctness check
#   ./run.sh pytorch bench         # run correctness check + attention benchmark
#   ./run.sh jax     bench         # run correctness check + attention benchmark
#   ./run.sh pytorch shell         # interactive shell with the GPU attached
#   ./run.sh jax     python x.py   # run an arbitrary command in the JAX image
#
# The docker run flags here are the Framework Desktop / ROCm essentials:
# device passthrough for /dev/kfd and /dev/dri, supplemental membership in the
# device nodes' actual numeric owner groups, and a relaxed seccomp profile.
#
# Env overrides:
#   ROCM_PYTORCH_TAG / ROCM_JAX_TAG  build a non-default base tag or tag@digest
#                                    (immutable default is in the Dockerfile ARG)
#   ROCM_ROOT=1                      run as root instead of your host user
#   WORKSPACE_DIR=<path>             mount a different host dir at /workspace
#                                    (default: ./workspace)
#   ROCM_CACHE_DIR=<path>            host dir persisted at $HOME/.cache inside
#                                    the container, so pip / MIOpen kernel
#                                    caches survive container exit
#                                    (default: ~/.cache/framework-rocm)
#   ROCM_HF_CACHE=<path>             host Hugging Face cache mounted at
#                                    $HOME/.cache/huggingface, shared with
#                                    native tools (default: ~/.cache/huggingface)
#   ROCM_PORTS="8888:8888 ..."       space-separated -p port mappings
#   HSA_OVERRIDE_GFX_VERSION=11.0.0  gfx fallback (see README), forwarded in
#   TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0
#                                    disable AOTriton mem-efficient SDPA.
#                                    Default ON: without it gfx1151 attention
#                                    falls back to the math backend (bf16 SDPA
#                                    ~11x slower, and the S×S materialization
#                                    OOMs training jobs). Verified faster on
#                                    torch 2.10 / ROCm 7.2.4; see
#                                    docs/gfx1151-attention-findings.md §6
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    0|false|FALSE|no|NO|off|OFF|"") return 1 ;;
    *)
      echo "ERROR: expected a boolean value (1/0, true/false, yes/no, on/off), got '$1'" >&2
      return 2
      ;;
  esac
}

# Print the unique numeric GIDs that actually own the GPU device nodes. Group
# names are not portable across distributions and often do not exist in the
# container, while Docker's --group-add reliably accepts the numeric owner.
device_gids() {
  local kfd_path="$1" dri_path="$2"
  local render_nodes=() node gid
  local -A seen=()

  if [ ! -e "${kfd_path}" ]; then
    echo "ERROR: ${kfd_path} does not exist; is amdgpu loaded?" >&2
    return 1
  fi
  if [ ! -d "${dri_path}" ]; then
    echo "ERROR: ${dri_path} does not exist; is amdgpu loaded?" >&2
    return 1
  fi
  shopt -s nullglob
  render_nodes=("${dri_path}"/renderD*)
  shopt -u nullglob
  if [ "${#render_nodes[@]}" -eq 0 ]; then
    echo "ERROR: no render nodes found under ${dri_path}" >&2
    return 1
  fi

  for node in "${kfd_path}" "${render_nodes[@]}"; do
    gid="$(stat -c '%g' "${node}")"
    if [[ ! "${gid}" =~ ^[0-9]+$ ]]; then
      echo "ERROR: could not determine numeric group owner for ${node}" >&2
      return 1
    fi
    if [[ -z "${seen[${gid}]+x}" ]]; then
      seen["${gid}"]=1
      printf '%s\n' "${gid}"
    fi
  done
}

# Append explicitly set host variables to a Docker argument array. Using -e
# only for variables that exist preserves an intentionally empty value while
# avoiding accidental overrides of image defaults.
append_forwarded_env() {
  local -n target="$1"
  shift
  local name
  for name in "$@"; do
    if [[ -v "${name}" ]]; then
      target+=(--env "${name}=${!name}")
    fi
  done
}

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
  local root_status=0
  if is_true "${ROCM_ROOT:-0}"; then
    :
  else
    root_status=$?
    if [ "${root_status}" -eq 2 ]; then
      return 2
    fi
    user_flags=(--user "$(id -u):$(id -g)")
  fi
  # Optional port mappings, e.g. ROCM_PORTS="8888:8888 6006:6006" for Jupyter
  # or TensorBoard running inside the container.
  local port_flags=()
  local p
  for p in ${ROCM_PORTS:-}; do
    port_flags+=(-p "$p")
  done
  # Persist caches across runs: HOME is set to /workspace (the mapped user has
  # no passwd entry, so it would otherwise be homeless), and a host dir is
  # mounted at /workspace/.cache — pip downloads and MIOpen's compiled-kernel
  # cache land under $HOME/.cache and survive --rm. Hugging Face models get the
  # host's *standard* HF cache mounted on top, so models are downloaded once
  # per machine and shared with native tools and other projects.
  # Pre-create both so Docker doesn't create them root-owned.
  local cache_dir="${ROCM_CACHE_DIR:-${HOME}/.cache/framework-rocm}"
  local hf_cache="${ROCM_HF_CACHE:-${HOME}/.cache/huggingface}"
  mkdir -p "${cache_dir}" "${hf_cache}"

  local gpu_group_flags=() gid gids
  if ! gids="$(device_gids /dev/kfd /dev/dri)"; then
    return 1
  fi
  while IFS= read -r gid; do
    [ -n "${gid}" ] || continue
    gpu_group_flags+=(--group-add "${gid}")
  done <<<"${gids}"
  if [ "${#gpu_group_flags[@]}" -eq 0 ]; then
    echo "ERROR: no GPU device group IDs were discovered" >&2
    return 1
  fi

  local forwarded_env=()
  append_forwarded_env forwarded_env \
    HSA_OVERRIDE_GFX_VERSION \
    EXPECTED_ARCH STRICT_ARCH EXPECTED_DEVICE STRICT_DEVICE \
    MAX_BF16_EFFICIENT_MS \
    PYTORCH_TUNABLEOP_ENABLED PYTORCH_TUNABLEOP_TUNING \
    PYTORCH_TUNABLEOP_FILENAME PYTORCH_TUNABLEOP_VERBOSE

  docker run --rm "${tty_flags[@]}" "${user_flags[@]}" "${port_flags[@]}" \
    --device=/dev/kfd \
    --device=/dev/dri \
    "${gpu_group_flags[@]}" \
    --security-opt seccomp=unconfined \
    --ipc=host \
    -e HOME=/workspace \
    -e "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}" \
    -e TRITON_CACHE_DIR=/workspace/.cache/triton \
    -e TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/inductor \
    "${forwarded_env[@]}" \
    -v "${WORKSPACE_DIR:-${HERE}/workspace}:/workspace" \
    -v "${cache_dir}:/workspace/.cache" \
    -v "${hf_cache}:/workspace/.cache/huggingface" \
    "${IMAGE}" "$@"
}

main() {
  local framework="${1:-pytorch}"
  shift || true
  local action="${1:-shell}"
  shift || true
  local image dockerfile tag_env check

  case "${framework}" in
    pytorch)
      image="framework-rocm:pytorch"
      dockerfile="Dockerfile"
      tag_env="ROCM_PYTORCH_TAG"   # canonical default lives in the Dockerfile ARG
      check="/usr/local/bin/check_gpu.py"
      ;;
    jax)
      image="framework-rocm:jax"
      dockerfile="Dockerfile.jax"
      tag_env="ROCM_JAX_TAG"       # canonical default lives in the Dockerfile ARG
      check="/usr/local/bin/check_jax.py"
      ;;
    *)
      echo "usage: $0 {pytorch|jax} {build|shell|check|bench|<command...>}" >&2
      return 2
      ;;
  esac

  # docker_run reads IMAGE so arbitrary commands and the named actions share
  # exactly the same runtime configuration.
  IMAGE="${image}"
  export IMAGE

  case "${action}" in
    build)
      # Pass --build-arg only when the tag is overridden; otherwise use the
      # Dockerfile ARG default (the single source of truth for the base tag).
      local build_args=() tag_val="${!tag_env:-}"
      if [ -n "${tag_val}" ]; then
        build_args=(--build-arg "${tag_env}=${tag_val}")
      fi
      docker build -f "${HERE}/${dockerfile}" "${build_args[@]}" -t "${image}" "${HERE}"
      ;;
    shell)
      docker_run /bin/bash
      ;;
    check)
      docker_run python "${check}"
      ;;
    bench)
      docker_run python "${check}" --bench
      ;;
    *)
      docker_run "${action}" "$@"
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
