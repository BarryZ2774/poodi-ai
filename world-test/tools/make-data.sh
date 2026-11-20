#!/bin/zsh
# Rebuilds the bundled San Francisco data: the base map, the building tiles and places.json.
# Needs: brew install pmtiles tippecanoe uv   (uv runs the overturemaps downloader)
# Takes about two minutes; the Overture download (about 150 MB) is kept in build/overture/.
set -euo pipefail
cd "$(dirname "$0")/.."
BBOX="-122.54,37.69,-122.34,37.86"
# Newest Protomaps daily planet build: see https://build-metadata.protomaps.dev/builds.json
BASE_URL="${BASE_URL:-https://build.protomaps.com/20261118.pmtiles}"
# Overture release the buildings come from. overturemaps 0.18.0 cannot pick a release by name
# (its release list is broken), so it downloads the latest one; this check stops if that is not this one.
OVERTURE_RELEASE="${OVERTURE_RELEASE:-2026-11-18.0}"
WORK=build/overture
mkdir -p WorldTest/Resources "$WORK"

pmtiles extract "$BASE_URL" WorldTest/Resources/sf-base.pmtiles --bbox=$BBOX

latest=$(curl -fsS https://stac.overturemaps.org/catalog.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["latest"])')
if [[ "$latest" != "$OVERTURE_RELEASE" ]]; then
  echo "Overture's latest release is $latest, not $OVERTURE_RELEASE. Set OVERTURE_RELEASE=$latest to use it." >&2
  exit 1
fi
for type in building building_part; do
  uvx overturemaps@0.18.0 download --bbox=$BBOX -f geojsonseq -t $type -o "$WORK/$type.geojsonseq"
  python3 -I tools/prepare_buildings.py "$WORK/$type.geojsonseq" "$WORK/$type.slim.geojsonseq" $type
done

# Every building at every zoom from 11 to 14 (no feature dropping, no tile size limit), so the city
# stays whole when zoomed out. Outlines are simplified to each zoom's tile resolution (tippecanoe's default).
tippecanoe -q -f -o WorldTest/Resources/sf-buildings.pmtiles -Z11 -z14 \
  --no-feature-limit --no-tile-size-limit --no-tiny-polygon-reduction \
  --clip-bounding-box=$BBOX \
  -n "Overture buildings $OVERTURE_RELEASE" \
  -A '<a href="https://www.openstreetmap.org/copyright">&copy; OpenStreetMap</a> <a href="https://docs.overturemaps.org/attribution">&copy; Overture Maps Foundation</a>' \
  -L building:"$WORK/building.slim.geojsonseq" \
  -L building_part:"$WORK/building_part.slim.geojsonseq"

python3 -I tools/make_places.py "$WORK/building.geojsonseq" "$WORK/building_part.geojsonseq"
ls -l WorldTest/Resources/*.pmtiles
