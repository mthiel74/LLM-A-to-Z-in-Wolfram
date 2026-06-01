#!/usr/bin/env bash
# wls-run.sh — run a wolframscript file with a hard timeout AND guaranteed kernel
# cleanup, so a long/stuck run can never leave an orphaned WolframKernel holding
# gigabytes of RAM.
#
# The problem this solves: `timeout N wolframscript -file foo.wls` only signals
# its direct child (the wolframscript wrapper). The actual WolframKernel is a
# *grandchild*; if the wrapper is killed (timeout, or a detached/backgrounded
# shell going away) the kernel can survive and run forever.
#
# Usage:  wls-run.sh [TIMEOUT_SECONDS] script.wls [args...]
#         wls-run.sh 200 chat_repl.wls nano30M_v3_sft.wlnet "What is 2+2?"
# If the first arg is not an integer, a default timeout of 600s is used.

set -uo pipefail

WOLFRAM="${WOLFRAMSCRIPT:-/Applications/Wolfram.app/Contents/MacOS/wolframscript}"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  TO="$1"; shift
else
  TO=600
fi

# Snapshot kernels that already existed, so we only reap the ones WE start.
before="$(pgrep -f 'WolframKernel' 2>/dev/null | sort || true)"

# Prefer GNU timeout (gtimeout) with --kill-after; fall back to plain timeout.
if command -v gtimeout >/dev/null 2>&1; then
  gtimeout --kill-after=10s --signal=TERM "$TO" "$WOLFRAM" -file "$@"
  rc=$?
elif command -v timeout >/dev/null 2>&1; then
  timeout --kill-after=10s --signal=TERM "$TO" "$WOLFRAM" -file "$@"
  rc=$?
else
  "$WOLFRAM" -file "$@" &
  wp=$!
  ( sleep "$TO"; kill -TERM "$wp" 2>/dev/null ) &
  wait "$wp"; rc=$?
fi

# Reap any kernel that we started and that is still alive (the orphan guard).
after="$(pgrep -f 'WolframKernel' 2>/dev/null | sort || true)"
new_kernels="$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") 2>/dev/null || true)"
if [[ -n "${new_kernels// /}" ]]; then
  echo "wls-run: reaping leftover kernel(s): ${new_kernels//$'\n'/ }" >&2
  # shellcheck disable=SC2086
  kill -9 $new_kernels 2>/dev/null || true
fi

exit "$rc"
