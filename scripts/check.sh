#!/usr/bin/env bash
# Static checks — no GPU and no image build required. Run before pushing; CI
# runs the same script (.github/workflows/checks.yml). Catches the regressions
# that don't need hardware: shell/python syntax, a valid compose config, and
# lint, and README reference drift vs the authoritative Dockerfile ARG defaults.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit

fail=0
note() { printf '  %s\n' "$*"; }
bad()  { printf '  ERROR: %s\n' "$*"; fail=1; }

echo "== bash syntax =="
if bash -n run.sh scripts/check.sh; then note "shell scripts OK"; else bad "shell syntax error"; fi

echo "== shellcheck =="
if ! command -v shellcheck >/dev/null 2>&1; then
  bad "shellcheck is unavailable — use the uv command in README.md#development"
elif shellcheck run.sh scripts/check.sh tests/test_run.sh; then
  note "ShellCheck OK"
else
  bad "ShellCheck failed"
fi

echo "== python compile =="
if python3 -m py_compile check_gpu.py check_jax.py tests/test_checks.py; then note "Python syntax OK"; else bad "python syntax error"; fi

echo "== ruff =="
if ! command -v ruff >/dev/null 2>&1; then
  bad "ruff is unavailable — use the uv command in README.md#development"
else
  if ruff check check_gpu.py check_jax.py tests; then note "Ruff lint OK"; else bad "Ruff lint failed"; fi
  if ruff format --check check_gpu.py check_jax.py tests; then note "Ruff format OK"; else bad "Ruff format failed"; fi
fi

echo "== hardware-free tests =="
if python3 -m unittest discover -s tests -p 'test_*.py'; then note "Python tests OK"; else bad "Python tests failed"; fi
if bash tests/test_run.sh; then note "shell tests OK"; else bad "shell tests failed"; fi

echo "== compose config (with .env.example) =="
if docker compose --env-file .env.example config >/dev/null; then note "compose config OK"; else bad "compose config invalid"; fi

echo "== compose runtime environment forwarding =="
compose_config="$(
  EXPECTED_ARCH=gfx-test STRICT_ARCH=1 MAX_BF16_EFFICIENT_MS=30 \
    PYTORCH_TUNABLEOP_ENABLED=1 \
    docker compose --env-file .env.example config 2>/dev/null
)"
for setting in \
  'EXPECTED_ARCH: gfx-test' \
  'STRICT_ARCH: "1"' \
  'MAX_BF16_EFFICIENT_MS: "30"' \
  'PYTORCH_TUNABLEOP_ENABLED: "1"'; do
  if grep -qF -- "${setting}" <<<"${compose_config}"; then
    note "compose forwards ${setting%%:*}"
  else
    bad "compose did not forward '${setting}'"
  fi
done

echo "== immutable defaults match README =="
check_image_ref() {
  local dockerfile="$1" arg="$2" ref
  ref="$(sed -n "s/^ARG ${arg}=//p" "${dockerfile}" | head -1)"
  if [ -z "${ref}" ]; then
    bad "no 'ARG ${arg}=' default found in ${dockerfile}"; return
  fi
  if [[ ! "${ref}" =~ @sha256:[0-9a-f]{64}$ ]]; then
    bad "${arg} default is not pinned by sha256 digest: ${ref}"
  elif grep -qF -- "${ref}" README.md; then
    note "${arg} immutable default present in README"
  else
    bad "${arg} default '${ref}' (${dockerfile}) is not mentioned in README.md — update the README"
  fi
}
check_image_ref Dockerfile     ROCM_PYTORCH_TAG
check_image_ref Dockerfile.jax ROCM_JAX_TAG

echo "== direct PyTorch dependencies are exactly constrained =="
while IFS= read -r requirement; do
  [[ -z "${requirement}" || "${requirement}" == \#* ]] && continue
  if [[ ! "${requirement}" =~ ^[^=]+==[^=]+$ ]]; then
    bad "requirements.txt entry is not exactly pinned: ${requirement}"
  elif grep -qFx -- "${requirement}" constraints-pytorch.txt; then
    note "constrained ${requirement%%==*}"
  else
    bad "requirements.txt entry missing from constraints-pytorch.txt: ${requirement}"
  fi
done < requirements.txt

echo
if [ "${fail}" -eq 0 ]; then echo "All static checks passed."; else echo "Static checks FAILED."; fi
exit "${fail}"
