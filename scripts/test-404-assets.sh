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

if 'href="assets/css/extended/custom.css"' in html:
    print("Expected 404.html to use a root-relative custom.css path, but found a relative href.")
    raise SystemExit(1)

if 'href="/assets/css/extended/custom.css"' not in html:
    print("Expected 404.html to include the root-relative custom.css href.")
    raise SystemExit(1)
PY

echo "PASS: 404.html uses a root-relative custom.css path."
