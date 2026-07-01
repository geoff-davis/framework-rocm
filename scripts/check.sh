#!/usr/bin/env bash
# Static checks — no GPU and no image build required. Run before pushing; CI
# runs the same script (.github/workflows/checks.yml). Catches the regressions
# that don't need hardware: shell/python syntax, a valid compose config, and
# README tag drift vs the authoritative Dockerfile ARG defaults.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

fail=0
note() { printf '  %s\n' "$*"; }
bad()  { printf '  ERROR: %s\n' "$*"; fail=1; }

echo "== bash syntax =="
if bash -n run.sh scripts/check.sh; then note "shell scripts OK"; else bad "shell syntax error"; fi

echo "== python compile =="
if python3 -m py_compile check_gpu.py check_jax.py; then note "check_*.py OK"; else bad "python syntax error"; fi

echo "== compose config (with .env.example) =="
if docker compose --env-file .env.example config >/dev/null; then note "compose config OK"; else bad "compose config invalid"; fi

echo "== README default tags match Dockerfile ARGs =="
check_tag() {
  local dockerfile="$1" arg="$2" tag
  tag="$(sed -n "s/^ARG ${arg}=//p" "${dockerfile}" | head -1)"
  if [ -z "${tag}" ]; then
    bad "no 'ARG ${arg}=' default found in ${dockerfile}"; return
  fi
  if grep -qF -- "${tag}" README.md; then
    note "${arg} default present in README (${tag})"
  else
    bad "${arg} default '${tag}' (${dockerfile}) is not mentioned in README.md — update the README"
  fi
}
check_tag Dockerfile     ROCM_PYTORCH_TAG
check_tag Dockerfile.jax ROCM_JAX_TAG

echo
if [ "${fail}" -eq 0 ]; then echo "All static checks passed."; else echo "Static checks FAILED."; fi
exit "${fail}"
