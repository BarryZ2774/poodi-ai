#!/bin/zsh
# Rebuilds the bundled San Francisco tiles. Needs: brew install pmtiles
set -euo pipefail
cd "$(dirname "$0")/.."
BBOX="--bbox=-122.54,37.69,-122.34,37.86"
# Newest Protomaps daily planet build: see https://build-metadata.protomaps.dev/builds.json
BASE_URL="${BASE_URL:-https://build.protomaps.com/20261118.pmtiles}"
# Overture buildings tiles, release 2026-11-18.0: see https://docs.overturemaps.org/examples/overture-tiles/
BUILDINGS_URL="${BUILDINGS_URL:-https://overturemaps-extras-us-west-2.s3.amazonaws.com/tiles/2026-11-18.0/buildings.pmtiles}"
mkdir -p WorldTest/Resources
pmtiles extract "$BASE_URL" WorldTest/Resources/sf-base.pmtiles $BBOX
pmtiles extract "$BUILDINGS_URL" WorldTest/Resources/sf-buildings.pmtiles $BBOX
echo "Now run: python3 tools/make_places.py"
