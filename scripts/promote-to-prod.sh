#!/usr/bin/env bash
# Promote the current dev `main` to the prod repo.
#
# Repo split:
#   dev  = pwaabdullah/pwaabdullah.github.io   → https://pwaabdullah.github.io/
#   prod = pwaabdullah/newabdullah.com         → https://www.newabdullah.com/
#
# Both repos use the same workflow (.github/workflows/hugo.yml) which builds
# Hugo with a dynamic baseURL from the Pages action output, so the same commit
# produces the right URLs in each environment.
#
# The only difference between repos is `static/CNAME` (binds the custom
# domain — must exist on prod, must NOT exist on dev). This script preserves
# the prod CNAME file automatically.
#
# Strategy: build a fresh prod branch from dev/main (so deletions and renames
# propagate cleanly), overlay the prod-only CNAME, force-push.
#
# Usage:
#   scripts/promote-to-prod.sh           # promote current HEAD to prod
#   scripts/promote-to-prod.sh --dry-run # show what would happen
#
# Requires:
#   - `prod` git remote pointing at git@github.com:pwaabdullah/newabdullah.com.git
#   - SSH/yubikey auth set up for github.com

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

cd "$(dirname "$0")/.."

if ! git remote get-url prod >/dev/null 2>&1; then
  echo "ERROR: no 'prod' remote configured. Run:"
  echo "  git remote add prod git@github.com:pwaabdullah/newabdullah.com.git"
  exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$BRANCH" != "main" ]]; then
  echo "ERROR: must be on main branch (currently on '$BRANCH')."
  exit 1
fi

if ! git diff-index --quiet HEAD --; then
  echo "ERROR: working tree has uncommitted tracked changes. Commit or stash first."
  exit 1
fi

# Stash untracked files (e.g. local-only drafts, scratch dirs) so they
# don't leak into the prod commit. Restored at exit (success or failure).
UNTRACKED=$(git ls-files --others --exclude-standard)
STASHED=0
cleanup() {
  # Always try to restore working tree state, even on error/Ctrl-C.
  if git rev-parse --verify --quiet main >/dev/null; then
    git checkout main --quiet 2>/dev/null || true
  fi
  if [[ -n "${TMP_BRANCH:-}" ]] && git rev-parse --verify --quiet "$TMP_BRANCH" >/dev/null 2>&1; then
    git branch -D "$TMP_BRANCH" --quiet 2>/dev/null || true
  fi
  if [[ "$STASHED" == "1" ]]; then
    git stash pop --quiet 2>/dev/null || echo "(warning: failed to restore stashed untracked files; check 'git stash list')"
  fi
}
trap cleanup EXIT

if [[ -n "$UNTRACKED" ]]; then
  echo "==> Stashing untracked files so they don't leak into prod:"
  echo "$UNTRACKED" | sed 's/^/    /'
  git stash push --include-untracked --quiet --message "promote-to-prod: stash untracked"
  STASHED=1
fi

echo "==> Fetching origin and prod"
git fetch origin --quiet
git fetch prod --quiet

LOCAL=$(git rev-parse HEAD)
ORIGIN=$(git rev-parse origin/main)
if [[ "$LOCAL" != "$ORIGIN" ]]; then
  echo "WARNING: local main ($LOCAL) is not equal to origin/main ($ORIGIN)."
  echo "         Push to origin first so dev and prod stay in sync."
  read -rp "         Continue anyway? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi

PROD_HEAD=$(git rev-parse prod/main)

echo ""
echo "==> Diff prod/main → local main (this is what prod will receive):"
git --no-pager diff "$PROD_HEAD".."$LOCAL" --stat | tail -25 || true
echo ""

# Strategy: build prod branch as a copy of main, then overlay prod-only files
# from the current prod/main (currently just static/CNAME).
TMP_BRANCH="_promote_$(date +%s)"
echo "==> Creating fresh promotion branch from main: $TMP_BRANCH"
git branch -f "$TMP_BRANCH" main >/dev/null
git checkout "$TMP_BRANCH" --quiet

# Restore prod-only files. List paths here that should only ever exist on
# prod and never in dev.
PROD_ONLY_PATHS=(
  "static/CNAME"
)
RESTORED_ANY=0
for path in "${PROD_ONLY_PATHS[@]}"; do
  if git show "prod/main:$path" >/dev/null 2>&1; then
    git show "prod/main:$path" > "$path"
    git add "$path"
    RESTORED_ANY=1
    echo "==> Restored prod-only file: $path"
  fi
done

if (( RESTORED_ANY )); then
  if ! git diff --cached --quiet; then
    git commit --quiet -m "Promote: $(git --no-pager log -1 --format='%s' main)

Promoted from dev (pwaabdullah.github.io) HEAD=$LOCAL.
Preserves prod-only files: ${PROD_ONLY_PATHS[*]}."
  fi
fi

# If the temp branch is identical to prod/main after CNAME overlay, nothing to do.
if [[ "$(git rev-parse "$TMP_BRANCH")" == "$(git rev-parse prod/main)" ]]; then
  echo "Nothing to promote — prod is already in sync with main (after CNAME overlay)."
  exit 0
fi

echo ""
echo "==> Commits on $TMP_BRANCH that prod/main doesn't have:"
git --no-pager log "$PROD_HEAD..$TMP_BRANCH" --oneline | head -10 || true
echo ""

if (( DRY_RUN )); then
  echo "==> DRY RUN — would force-push $TMP_BRANCH → prod main."
  echo "(no push performed)"
  exit 0
fi

echo "==> Force-pushing $TMP_BRANCH → prod main (this overwrites prod/main)"
git push prod "$TMP_BRANCH:main" --force

echo ""
echo "==> Promotion pushed. Watch the deploy:"
echo "    gh run watch \$(gh run list --repo pwaabdullah/newabdullah.com --limit 1 --json databaseId --jq '.[0].databaseId') --repo pwaabdullah/newabdullah.com --exit-status"
echo ""
echo "==> Once deployed, prod will be live at:"
echo "    https://www.newabdullah.com/"
