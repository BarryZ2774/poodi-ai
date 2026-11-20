# World Test

A small iOS test app (SwiftUI, Swift 6, iOS 26+) that shows San Francisco as a pale 3D city in two map engines, to compare them by eye and for smoothness on a real phone.

- **Ours:** MapLibre Native 6.31.0 with our own "porcelain" style (`WorldTest/Resources/porcelain-style.json`) over PMTiles bundled in the app: an OpenStreetMap base map (Protomaps daily build 2026-11-18) and Overture Maps buildings (release 2026-11-18.0). No server needed.
- **Mapbox:** Mapbox Maps SDK 11.35.1 with the Standard style, theme `faded`, light preset `day`, 3D objects and landmarks on. Needs a public token in `Config/Secrets.xcconfig` (see `MORNING.md`).

To run it on a phone, see **MORNING.md**. Screenshots from the simulator are in `shots/`.

## Layout

| Path | What |
| --- | --- |
| `project.yml` | XcodeGen spec. `xcodegen generate` rebuilds `WorldTest.xcodeproj`. |
| `WorldTest/Sources` | SwiftUI app: shared camera model, the two map views, fps/memory monitor. |
| `WorldTest/Resources` | Style, PMTiles, glyphs (Noto Sans, SIL OFL), `places.json`. |
| `Config/` | `App.xcconfig`; `Secrets.xcconfig` (git-ignored) holds the Mapbox token. |
| `tools/make-data.sh` | Re-extracts the two San Francisco PMTiles files (needs `brew install pmtiles`). |
| `tools/make_places.py` | Rebuilds `places.json`, the sample landmarks for "Colour my places" (needs `tippecanoe`). |
| `tools/shoot.sh` | Builds, installs and screenshots the Ours tab on a simulator. |

## Notes

- **Buildings:** fill-extrusion from Overture `height`, else `num_floors × 3.2 m`, else 6 m. Building parts are drawn with their own `height`/`min_height`, and the plain outline of any building with parts is hidden, so the Transamerica Pyramid, the Ferry Building clock tower and the Golden Gate Bridge towers show their real shapes.
- **Colour my places (Ours):** MapLibre feature state (`MLNVectorTileSource.setFeatureState`, new in 6.30) on the `building` and `building_part` layers. Feature state is keyed on the numeric vector-tile feature id: `promoteId` on the PMTiles source had no effect in MapLibre Native 6.31, so `places.json` stores those numeric ids (from the zoom 14 tiles; the 3D view always uses zoom 14 tiles). States are set only after the first frame, because states set earlier are dropped. If the API refuses, the app falls back to two filtered terracotta layers. Test the fallback with the launch argument `-forceFallback YES`.
- **Colour my places (Mapbox):** the Standard style's `select` state on the `standardBuildings` featureset, coloured with `colorBuildingSelect`. Mapbox building ids are not known ahead of time, so each landmark on screen is found with a rendered-feature query at its coordinate. This re-runs whenever the map comes to rest. Not tested yet: no token on the build Mac.
- **Overture's low-zoom tiles** leave out most buildings, so the city looks sparse when zoomed out further than about zoom 13.5.
- **Launch arguments** (Xcode → Scheme → Run → Arguments): `-preset downtown|bridge|top`, `-engine ours|mapbox`, `-colour YES`, `-autoTour YES` (loops through the presets and logs fps to the console), `-forceFallback YES`.

## Data and credits

© OpenStreetMap contributors (ODbL), via Protomaps. © Overture Maps Foundation (buildings: ODbL/CDLA, see docs.overturemaps.org/attribution). Noto Sans glyphs from protomaps/basemaps-assets, SIL Open Font License (`WorldTest/Resources/glyphs/OFL.txt`).
