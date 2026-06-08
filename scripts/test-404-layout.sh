#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

cd "$repo_root"
hugo --cleanDestinationDir --destination "$build_dir" >/dev/null

python3 - "$build_dir/404.html" <<'PY'
from pathlib import Path
import sys

html = Path(sys.argv[1]).read_text()

if 'class="not-found"' in html:
    print('Expected custom 404 markup to avoid the theme\'s .not-found class collision.')
    raise SystemExit(1)

if 'class="nf-page"' not in html:
    print('Expected custom 404 markup to use the nf-page wrapper class.')
    raise SystemExit(1)
PY

echo "PASS: 404 markup avoids the theme .not-found class collision."
