#!/usr/bin/env bash
# Fail if the required test suite fails OR silently skips a test.
#
# GitHub's default run shell does not guarantee pipefail. Without it, tee can
# mask pytest's non-zero exit code and make a broken CI job look successful.
set -euo pipefail

log="$(mktemp)"
trap 'rm -f "$log"' EXIT

# Keep pytest output visible in Actions logs while preserving its exit status.
pytest -rs "$@" | tee "$log"

# These required integration tests use pytest.importorskip, so a missing
# dependency must fail this CI job rather than being treated as a green build.
if grep -q "SKIPPED" "$log"; then
    echo "ERROR: Required pytest tests were skipped." >&2
    exit 1
fi
