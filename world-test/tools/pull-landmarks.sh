#!/bin/zsh
# Copies every landmark model from the remote branches model/<slug> into the app, as data only.
#
#   tools/pull-landmarks.sh
#
# For each branch origin/model/<slug> it copies landmarks/models/<slug>/<slug>.glb and facts.json
# with `git show` into WorldTest/Resources/landmarks/<slug>/, writes landmarks/index.json
# (slug, name, kind, anchor, commit) and then works out which of the map's own buildings and bridge
# roads each model stands on (tools/landmark_overlap.py), so the app can hide them.
# Nothing from those branches is run: their build scripts are never copied or executed.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=WorldTest/Resources/landmarks
mkdir -p "$OUT"

git fetch --quiet origin '+refs/heads/model/*:refs/remotes/origin/model/*'

entries=()
skipped=()
for ref in $(git for-each-ref --format='%(refname:short)' 'refs/remotes/origin/model/'); do
  slug=${ref#origin/model/}
  dir="landmarks/models/$slug"
  commit=$(git rev-parse --short "$ref")
  if ! git cat-file -e "$ref:$dir/$slug.glb" 2>/dev/null || ! git cat-file -e "$ref:$dir/facts.json" 2>/dev/null; then
    skipped+=("$slug (branch $commit has no $slug.glb or facts.json)")
    continue
  fi
  tmp=$(mktemp -d)
  git show "$ref:$dir/$slug.glb" > "$tmp/$slug.glb"
  git show "$ref:$dir/facts.json" > "$tmp/facts.json"
  # facts.json must have a name and an anchor; read it as data with jq-free plain Python.
  if ! entry=$(python3 -I - "$tmp/facts.json" "$slug" "$commit" <<'PY'
import json, sys
path, slug, commit = sys.argv[1:4]
f = json.load(open(path))
a = f["anchor"]
print(json.dumps({"slug": slug, "name": f.get("name") or slug, "kind": f.get("kind", "building"),
                  "anchor": {"lat": float(a["lat"]), "lon": float(a["lon"])}, "commit": commit}))
PY
  ); then
    skipped+=("$slug (facts.json at $commit has no name or anchor)")
    rm -rf "$tmp"
    continue
  fi
  rm -rf "$OUT/$slug"
  mkdir -p "$OUT/$slug"
  mv "$tmp/$slug.glb" "$tmp/facts.json" "$OUT/$slug/"
  rm -rf "$tmp"
  entries+=("$entry")
  echo "copied $slug from $commit"
done

# Folders left over from branches that no longer exist are removed.
for folder in "$OUT"/*(N/); do
  slug=${folder:t}
  git show-ref --quiet "refs/remotes/origin/model/$slug" || { rm -rf "$folder"; echo "removed $slug (no branch)"; }
done

printf '%s\n' "${entries[@]}" | python3 -I -c '
import json, sys
items = [json.loads(line) for line in sys.stdin if line.strip()]
json.dump(sorted(items, key=lambda i: i["name"]), open(sys.argv[1], "w"), indent=2)
' "$OUT/index.json"
echo "wrote $OUT/index.json (${#entries[@]} landmarks)"
for s in "${skipped[@]}"; do echo "skipped $s"; done

# Which map buildings and bridge roads lie under each model (needs tippecanoe-decode).
python3 -I tools/landmark_overlap.py WorldTest/Resources
