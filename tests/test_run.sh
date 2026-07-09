#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./run.sh

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

for value in 1 true YES on; do
  is_true "${value}" || fail "expected '${value}' to be true"
done
for value in 0 false NO off ""; do
  if is_true "${value}"; then
    fail "expected '${value}' to be false"
  fi
done
if is_true sometimes 2>/dev/null; then
  fail "invalid boolean unexpectedly succeeded"
else
  status=$?
  [ "${status}" -eq 2 ] || fail "invalid boolean returned ${status}, expected 2"
fi

unset EXPECTED_ARCH STRICT_ARCH MAX_BF16_EFFICIENT_MS PYTORCH_TUNABLEOP_ENABLED
export EXPECTED_ARCH=""
export MAX_BF16_EFFICIENT_MS=30
export PYTORCH_TUNABLEOP_ENABLED=1
env_args=()
append_forwarded_env env_args \
  EXPECTED_ARCH STRICT_ARCH MAX_BF16_EFFICIENT_MS PYTORCH_TUNABLEOP_ENABLED
expected_args=(
  --env "EXPECTED_ARCH="
  --env "MAX_BF16_EFFICIENT_MS=30"
  --env "PYTORCH_TUNABLEOP_ENABLED=1"
)
[ "${env_args[*]}" = "${expected_args[*]}" ] || \
  fail "forwarded env mismatch: ${env_args[*]}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT
mkdir -p "${tmpdir}/dri"
touch "${tmpdir}/kfd" "${tmpdir}/dri/renderD128" "${tmpdir}/dri/renderD129"
mapfile -t gids < <(device_gids "${tmpdir}/kfd" "${tmpdir}/dri")
[ "${#gids[@]}" -eq 1 ] || fail "expected duplicate GIDs to be de-duplicated"
[ "${gids[0]}" = "$(id -g)" ] || fail "device GID did not match actual file owner"

if device_gids "${tmpdir}/missing" "${tmpdir}/dri" >/dev/null 2>&1; then
  fail "missing KFD device unexpectedly succeeded"
fi

echo "run.sh helpers OK"
