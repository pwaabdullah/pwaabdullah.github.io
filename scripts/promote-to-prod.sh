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
# The only thing prod has that dev doesn't is `static/CNAME` (which binds the
# custom domain). That file lives only in the prod repo and we never touch it
# from here — we promote main except for that one file.
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
  echo "ERROR: working tree is dirty. Commit or stash first."
  exit 1
fi

echo "==> Fetching latest from origin and prod"
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
echo "==> Diff prod/main → local main:"
git log --oneline "$PROD_HEAD..$LOCAL" || true
echo ""

# Build a temporary "prod" commit on top that preserves prod's static/CNAME.
# We don't carry static/CNAME locally (dev doesn't bind the custom domain),
# but the prod repo has it. The cleanest way to promote without dropping it
# is to merge our changes on top of prod/main, keeping prod's CNAME file.
TMP_BRANCH="_promote_$(date +%s)"
echo "==> Creating temporary promotion branch: $TMP_BRANCH (based on prod/main)"
git branch -f "$TMP_BRANCH" prod/main >/dev/null

# Apply our commits on top, but preserve any files that exist only on prod
# (notably static/CNAME). We do this by checking out our tree into prod's
# branch as a single commit, then restoring static/CNAME from prod/main.
git checkout "$TMP_BRANCH" --quiet
git checkout main -- . >/dev/null

# Restore prod-only files (currently just static/CNAME).
if git show "prod/main:static/CNAME" >/dev/null 2>&1; then
  git checkout prod/main -- static/CNAME
fi

if git diff --cached --quiet && git diff --quiet; then
  echo "Nothing to promote — prod is already up to date with local main (after CNAME merge)."
  git checkout main --quiet
  git branch -D "$TMP_BRANCH" >/dev/null
  exit 0
fi

git add -A
git commit -m "Promote: $(git --no-pager log -1 --format='%s' main)

Promoted from dev (pwaabdullah.github.io) HEAD=$LOCAL.
Preserves prod-only files (static/CNAME).
"

if (( DRY_RUN )); then
  echo ""
  echo "==> DRY RUN — would push $TMP_BRANCH to prod/main:"
  git --no-pager log prod/main..$TMP_BRANCH --oneline
  git checkout main --quiet
  git branch -D "$TMP_BRANCH" >/dev/null
  echo "(no push performed)"
  exit 0
fi

echo ""
echo "==> Pushing $TMP_BRANCH → prod main"
git push prod "$TMP_BRANCH:main"

git checkout main --quiet
git branch -D "$TMP_BRANCH" >/dev/null

echo ""
echo "==> Promotion pushed. Watch the deploy:"
echo "    gh run watch \$(gh run list --repo pwaabdullah/newabdullah.com --limit 1 --json databaseId --jq '.[0].databaseId') --repo pwaabdullah/newabdullah.com --exit-status"
echo ""
echo "==> Once deployed, prod will be live at:"
echo "    https://www.newabdullah.com/"
