#!/usr/bin/env bash
# Apply local opendbc patches to a checkout of opendbc.
#
# Intended to run against openpilot's opendbc_repo submodule before the build,
# so a stock comma branch can be tracked while still carrying local fixes.
#
# Usage: ./apply.sh [path-to-opendbc-checkout]   (default: openpilot's opendbc_repo)
#
# Fails loudly if a patch does not apply. That is deliberate: a silently
# skipped patch means shipping stock behavior while believing the fix is in.

set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$PATCH_DIR/../opendbc_repo}"

if [ ! -d "$TARGET/opendbc" ]; then
  echo "ERROR: '$TARGET' does not look like an opendbc checkout." >&2
  exit 1
fi

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
if [ ${#patches[@]} -eq 0 ]; then
  echo "ERROR: no .patch files found in $PATCH_DIR" >&2
  exit 1
fi

for p in "${patches[@]}"; do
  name="$(basename "$p")"

  # Already applied (e.g. rebuild without a clean checkout) -> not an error.
  if git -C "$TARGET" apply --reverse --check "$p" 2>/dev/null; then
    echo "already applied: $name"
    continue
  fi

  if ! git -C "$TARGET" apply --check "$p" 2>/dev/null; then
    echo "" >&2
    echo "ERROR: patch does not apply: $name" >&2
    echo "" >&2
    echo "Upstream opendbc has likely changed the code this patch touches." >&2
    echo "Refresh the patch against the current opendbc before building:" >&2
    echo "  cd $TARGET && git apply --3way $p   # then resolve, and re-export" >&2
    echo "" >&2
    exit 1
  fi

  git -C "$TARGET" apply "$p"
  echo "applied: $name"
done

# Regenerate the DBC files, since the patches touch generator sources.
echo "regenerating DBCs..."
PYTHONPATH="$TARGET" python3 "$TARGET/opendbc/dbc/generator/generator.py"

echo "all patches applied"
