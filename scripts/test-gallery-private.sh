#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

cd "$repo_root"

# Reproduce the stale-output case that bit /gallery/: the destination
# already contains an old page, and the next build must remove it.
mkdir -p "$build_dir/gallery"
printf 'stale gallery output\n' > "$build_dir/gallery/index.html"

hugo --cleanDestinationDir --destination "$build_dir" >/dev/null

if [[ -e "$build_dir/gallery/index.html" ]]; then
  echo "Expected gallery page to stay private and remove stale output, but /gallery/ still exists."
  exit 1
fi

echo "PASS: gallery page is not rendered and stale output is removed."
