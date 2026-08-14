#!/usr/bin/env bash
# Origin canonical verifier (docs/SPEC.md §12).
#
# Runs every required check and exits non-zero if ANY fails. Missing required
# components are FAILURES, not skips (V-2): this script must only exit 0 when
# the project is genuinely complete per docs/SPEC.md.
#
# Usage: bash scripts/verify.sh

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Offline/deterministic environment for all Python steps (Q-2/V-1): unit tests
# and checks must never download models or touch the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

STEPS=()
RESULTS=()
FAILED=0

run_step() {
  local name="$1"; shift
  echo ""
  echo "━━━ ${name} ━━━"
  if "$@"; then
    STEPS+=("$name"); RESULTS+=("PASS")
  else
    STEPS+=("$name"); RESULTS+=("FAIL")
    FAILED=1
  fi
}

fail_step() {
  local name="$1" reason="$2"
  echo ""
  echo "━━━ ${name} ━━━"
  echo "FAIL: ${reason}"
  STEPS+=("$name"); RESULTS+=("FAIL")
  FAILED=1
}

require_file() { # require_file <step-name> <path> — fail the step if path missing
  local name="$1" path="$2"
  if [[ ! -e "$path" ]]; then
    fail_step "$name" "required path '$path' does not exist yet"
    return 1
  fi
  return 0
}

# ─── Python toolchain ────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
  fail_step "toolchain: uv" "uv is not installed (https://docs.astral.sh/uv/)"
else
  STEPS+=("toolchain: uv"); RESULTS+=("PASS")
fi

if require_file "python: project" "pyproject.toml"; then
  run_step "python: uv sync (frozen)" uv sync --frozen

  run_step "python: ruff lint"         uv run ruff check .
  run_step "python: ruff format"       uv run ruff format --check .
  run_step "python: mypy"              uv run mypy .
  run_step "python: import sanity"     uv run python -c "import origin_ml, origin_api; print('imports ok:', origin_ml.__name__, origin_api.__name__)"
  run_step "python: pytest (offline)"  uv run pytest -q
fi

# ─── Backend smoke (in-process, stub detectors, no downloads) ────────────────
if require_file "backend: smoke" "scripts/smoke_api.py"; then
  run_step "backend: smoke" uv run python scripts/smoke_api.py
fi

# ─── Frontend ────────────────────────────────────────────────────────────────
if require_file "web: project" "apps/web/package.json"; then
  pushd apps/web >/dev/null

  if [[ -f package-lock.json ]]; then
    run_step "web: npm ci" npm ci --no-audit --no-fund
  else
    fail_step "web: npm ci" "package-lock.json missing (R-1: lockfile must be committed)"
  fi

  run_step "web: eslint"       npm run --silent lint
  run_step "web: vitest"       npm run --silent test
  run_step "web: prod build"   npm run --silent build

  popd >/dev/null
fi

# ─── Summary (V-3) ───────────────────────────────────────────────────────────
echo ""
echo "══════════════════════ VERIFY SUMMARY ══════════════════════"
for i in "${!STEPS[@]}"; do
  printf '  %-28s %s\n' "${STEPS[$i]}" "${RESULTS[$i]}"
done
echo "════════════════════════════════════════════════════════════"
if [[ "$FAILED" -ne 0 ]]; then
  echo "OVERALL: FAIL"
  exit 1
fi
echo "OVERALL: PASS"
exit 0
