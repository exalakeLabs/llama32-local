#!/usr/bin/env zsh
set -euo pipefail

ROOT="${0:A:h}"
cd "$ROOT"

if [[ -f "$ROOT/.runtime" ]]; then
  if ! source "$ROOT/.runtime" >/dev/null 2>/dev/null; then
    set -a
    [[ -f "$ROOT/.env.default" ]] && source "$ROOT/.env.default"
    [[ -f "$ROOT/.env" ]] && source "$ROOT/.env"
    set +a
  fi
else
  set -a
  [[ -f "$ROOT/.env.default" ]] && source "$ROOT/.env.default"
  [[ -f "$ROOT/.env" ]] && source "$ROOT/.env"
  set +a
fi

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

exec "$PYTHON" "$ROOT/src/layoutlm_invoices.py" "$@"
